"""
Fetches the full source of methods/functions that contain changed diff lines.

Flow:
  1. Fetch the file blob from GitHub (using blob_sha already in FileDiff)
  2. Parse the diff patch to find which new-file line numbers were changed
  3. For each changed line, find the enclosing method using language-specific heuristics:
     - Java/Kotlin/Go/JS/TS: brace-depth counting
     - Python: indentation tracking
  4. Deduplicate overlapping method ranges
  5. Return MethodContext objects attached to the FileDiff

No new dependencies — same GitHub blob API pattern as pr_fetcher._fetch_gitignore_spec.
"""

import base64
import logging
import re
from dataclasses import dataclass

import httpx

from app.models.review import FileDiff, MethodContext

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Cap a single method at this many lines to prevent huge context blocks
MAX_METHOD_LINES = 120
# Total method context lines across all files in one PR
MAX_TOTAL_METHOD_LINES = 600


# ── blob fetcher ──────────────────────────────────────────────────────────────

async def _fetch_file_lines(
    client: httpx.AsyncClient,
    repo: str,
    path: str,
    headers: dict,
) -> list[str] | None:
    """
    Fetch file content via the contents API (default branch).
    Uses the same API and permissions as the .gitignore fetch (Contents: Read).
    For method context extraction, the default branch is accurate enough —
    method structure is stable across PR branches.
    """
    try:
        resp = await client.get(
            f"{GITHUB_API}/repos/{repo}/contents/{path}",
            headers=headers,
        )
        if resp.status_code == 404:
            logger.debug("File %s not found in %s", path, repo)
            return None
        if resp.status_code == 403:
            logger.debug(
                "Contents: Read permission not granted for %s — skipping method context. "
                "Grant the GitHub App 'Contents: Read' in App settings to enable this feature.",
                repo,
            )
            return None
        resp.raise_for_status()
        data = resp.json()
        content = base64.b64decode(data["content"].replace("\n", "")).decode("utf-8", errors="replace")
        return content.splitlines()
    except Exception as exc:
        logger.debug("Failed to fetch %s from %s: %s", path, repo, exc)
        return None


# ── diff line parser ──────────────────────────────────────────────────────────

def _changed_lines_from_patch(patch: str) -> set[int]:
    """
    Return the set of new-file line numbers that were added or modified
    (context + added lines, not deleted). These are the lines we want
    to find the enclosing method for.
    """
    lines: set[int] = set()
    current = 0
    for raw in patch.splitlines():
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if hunk:
            current = int(hunk.group(1)) - 1
            continue
        if raw.startswith("-"):
            continue
        current += 1
        if raw.startswith("+"):   # only added lines, not context
            lines.add(current)
    return lines


# ── method boundary finders ───────────────────────────────────────────────────

def _find_method_brace_based(lines: list[str], target: int) -> tuple[int, int] | None:
    """
    Find method boundaries using brace-depth counting (Java/Kotlin/Go/JS/TS).

    Strategy:
      - Walk backward from target tracking brace depth.
      - The method starts at the line where depth returns to 0 from 1 (the
        opening '{' of the method body), then walk back further past the
        multi-line signature and annotations.
      - Walk forward from that '{' to find the matching closing '}'.

    Returns (start_line, end_line) as 1-indexed line numbers, or None.
    """
    n = len(lines)
    t = target - 1  # convert to 0-indexed

    if t < 0 or t >= n:
        return None

    # Walk backward to find the opening '{' of the enclosing method
    depth = 0
    method_open_idx = None
    for i in range(t, -1, -1):
        depth += lines[i].count("}") - lines[i].count("{")
        if depth < 0:
            # We've gone above the enclosing block — this is the method open
            method_open_idx = i
            break

    if method_open_idx is None:
        return None

    # Walk further back past annotations (@Override, @Bean, etc.) and the
    # multi-line method signature to find the true start of the declaration
    sig_start = method_open_idx
    for i in range(method_open_idx - 1, max(method_open_idx - 15, -1), -1):
        stripped = lines[i].strip()
        if not stripped:
            break
        # Stop at blank lines, closing braces (end of previous method), or
        # lines that look like the end of the *previous* method's body
        if stripped == "}" or stripped.startswith("//") and i < method_open_idx - 2:
            break
        sig_start = i

    # Walk forward from method_open_idx to find the closing '}'
    depth = 0
    method_close_idx = None
    for i in range(method_open_idx, n):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > method_open_idx:
            method_close_idx = i
            break

    if method_close_idx is None:
        return None

    return sig_start + 1, method_close_idx + 1  # back to 1-indexed


def _find_method_python(lines: list[str], target: int) -> tuple[int, int] | None:
    """
    Find method boundaries using indentation tracking (Python).

    Walk backward to find the nearest 'def ' at indentation ≤ target line's
    indentation minus one level. Walk forward until indentation returns to
    that level or lower (end of function body).
    """
    n = len(lines)
    t = target - 1  # 0-indexed

    if t < 0 or t >= n:
        return None

    target_indent = len(lines[t]) - len(lines[t].lstrip())

    # Walk backward to find 'def'
    def_idx = None
    for i in range(t, -1, -1):
        stripped = lines[i].lstrip()
        indent = len(lines[i]) - len(stripped)
        if stripped.startswith("def ") and indent < target_indent:
            def_idx = i
            break
        if indent < target_indent and stripped and not stripped.startswith("#"):
            # Hit a line at lower indentation that isn't a def — stop
            break

    if def_idx is None:
        return None

    def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip())

    # Walk forward until we hit a non-empty, non-comment line at ≤ def_indent
    # that isn't the def line itself
    end_idx = n - 1
    for i in range(def_idx + 1, n):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[i]) - len(stripped)
        if indent <= def_indent:
            end_idx = i - 1
            break

    return def_idx + 1, end_idx + 1  # 1-indexed


def _find_boundaries(lines: list[str], target: int, lang: str) -> tuple[int, int] | None:
    """Dispatch to the right boundary finder based on language."""
    if lang == "Python":
        return _find_method_python(lines, target)
    # Java, Kotlin, Go, TypeScript, JavaScript all use brace-depth
    return _find_method_brace_based(lines, target)


# ── public entry point ────────────────────────────────────────────────────────

async def extract_method_contexts(
    client: httpx.AsyncClient,
    repo: str,
    head_sha: str,
    headers: dict,
    diff_file: FileDiff,
    lang: str,
) -> list[MethodContext]:
    """
    For a single FileDiff, fetch the file via the contents API and
    extract MethodContext objects for each distinct method containing a changed line.
    """
    if not diff_file.patch:
        return []

    file_lines = await _fetch_file_lines(client, repo, diff_file.filename, headers)
    if file_lines is None:
        return []

    changed = _changed_lines_from_patch(diff_file.patch)
    if not changed:
        return []

    # Find method boundary for each changed line, deduplicate by range
    seen_ranges: set[tuple[int, int]] = set()
    contexts: list[MethodContext] = []

    for target_line in sorted(changed):
        bounds = _find_boundaries(file_lines, target_line, lang)
        if bounds is None:
            continue
        start, end = bounds
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))

        method_lines = file_lines[start - 1 : end]
        truncated = False
        if len(method_lines) > MAX_METHOD_LINES:
            method_lines = method_lines[:MAX_METHOD_LINES]
            truncated = True

        contexts.append(MethodContext(
            start_line=start,
            end_line=end,
            source="\n".join(method_lines),
            truncated=truncated,
        ))

    logger.info(
        "%s — extracted %d method context(s) from %d changed lines",
        diff_file.filename, len(contexts), len(changed),
    )
    return contexts
