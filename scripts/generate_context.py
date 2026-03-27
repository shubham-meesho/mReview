#!/usr/bin/env python3
"""
generate_context.py — Bootstrap .mreview/ context files from historical PR review comments.

Fetches the last N closed PRs, collects human review comments, sends them to Claude
for pattern extraction, then writes the results as .mreview/ markdown files in the repo.

Usage:
    python scripts/generate_context.py --repo Meesho/product-amplifyr
    python scripts/generate_context.py --repo Meesho/product-amplifyr --pr-count 100
    python scripts/generate_context.py --repo Meesho/product-amplifyr --dry-run
"""

import argparse
import asyncio
import base64
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx

# Allow imports from project root (app.*)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default local output dir: mreview-context/{repo_name}/
_LOCAL_CONTEXT_ROOT = Path(__file__).parent.parent / "mreview-context"

from app.github.app_auth import get_installation_token  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Comments shorter than this are almost certainly reactions, not review feedback
_MIN_COMMENT_CHARS = 20

# Prefixes/exact strings that indicate a trivial reaction rather than a real review
_TRIVIAL_PREFIXES = (
    "lgtm", "looks good", "+1", "nit:", "approved", "done", "thanks",
    "thank you", "nice", "good catch", "agreed", "yes", "no,", "ok,", "ok.",
)


def _is_trivial(body: str) -> bool:
    lower = body.lower().strip()
    if len(lower) < _MIN_COMMENT_CHARS:
        return True
    return any(lower.startswith(p) for p in _TRIVIAL_PREFIXES)


def _is_bot(login: str) -> bool:
    return "[bot]" in login or login.lower().endswith("bot")


# ── GitHub fetchers ────────────────────────────────────────────────────────────

async def _fetch_closed_prs(
    client: httpx.AsyncClient, repo: str, headers: dict, count: int
) -> list[dict]:
    prs: list[dict] = []
    page = 1
    while len(prs) < count:
        resp = await client.get(
            f"{GITHUB_API}/repos/{repo}/pulls",
            headers=headers,
            params={
                "state": "closed",
                "per_page": min(count - len(prs), 100),
                "page": page,
                "sort": "updated",
                "direction": "desc",
            },
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        prs.extend(batch)
        page += 1
    return prs[:count]


async def _fetch_inline_comments(
    client: httpx.AsyncClient, repo: str, pr_number: int, headers: dict
) -> list[dict]:
    """Line-level review comments (the ones attached to specific diff lines)."""
    resp = await client.get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/comments",
        headers=headers,
        params={"per_page": 100},
    )
    return resp.json() if resp.status_code == 200 else []


async def _fetch_review_bodies(
    client: httpx.AsyncClient, repo: str, pr_number: int, headers: dict
) -> list[dict]:
    """PR-level review summaries (CHANGES_REQUESTED / COMMENTED with a body)."""
    resp = await client.get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews",
        headers=headers,
        params={"per_page": 100},
    )
    return resp.json() if resp.status_code == 200 else []


# ── Comment filtering ──────────────────────────────────────────────────────────

def _collect_substantive_comments(
    pr: dict,
    inline: list[dict],
    reviews: list[dict],
) -> dict | None:
    """
    Filter a PR's comments down to substantive review feedback.
    Returns None if no useful comments remain.
    """
    comments: list[dict] = []

    for c in inline:
        if _is_bot(c["user"]["login"]) or _is_trivial(c["body"]):
            continue
        comments.append({
            "body": c["body"].strip(),
            "file": c.get("path"),
            "line": c.get("original_line"),
        })

    for r in reviews:
        body = (r.get("body") or "").strip()
        if not body:
            continue
        if _is_bot(r["user"]["login"]) or _is_trivial(body):
            continue
        # Only include CHANGES_REQUESTED and substantive COMMENTED reviews
        if r.get("state") not in ("CHANGES_REQUESTED", "COMMENTED"):
            continue
        comments.append({"body": body})

    if not comments:
        return None

    return {
        "pr_number": pr["number"],
        "pr_title": pr["title"],
        "comments": comments,
    }


# ── Prompt building ────────────────────────────────────────────────────────────

def _build_comments_text(pr_data: list[dict]) -> str:
    lines: list[str] = []
    for entry in pr_data:
        lines.append(f"\n--- PR #{entry['pr_number']}: {entry['pr_title']} ---")
        for c in entry["comments"]:
            loc = f" [{c['file']}:{c['line']}]" if c.get("file") and c.get("line") else ""
            lines.append(f"  REVIEW{loc}: {c['body']}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are a senior engineering lead analyzing code review history for a software team.
Your job is to extract durable, reusable patterns from these historical review comments
to guide future automated code reviews for this specific repository.

Output EXACTLY four sections with these exact markdown headers. No other text outside
the sections.

## Recurring Review Flags
Patterns that appeared across multiple PRs.
Format: "- When [situation], always/never [action]. (Observed in N PRs)"
Only include patterns observed in 2 or more PRs.
Be specific to this codebase — reference class names, patterns, or frameworks you observe.

## Team Anti-Patterns
Code patterns this team consistently rejects.
Format: "- Avoid [specific pattern] because [reason]."
Only include patterns that are team/codebase-specific, not generic best practices every team follows.

## Architecture & Constraints
Architectural rules and constraints implied by the review feedback
(layering rules, service ownership, naming conventions, forbidden dependencies).
Format: "- [Rule]"

## Past Incident Patterns
Comments that suggest past production pain points, near-misses, or classes of bugs
the team has been burned by (performance issues, data corruption, deployment failures,
race conditions, serialization breakage, etc.).
Format: "- [Pattern]: [what went wrong or could go wrong]"

Rules:
- Be specific, not generic. "Avoid calling ProductService.getDetails() in a loop" beats "avoid N+1"
- Omit all reviewer names
- Skip one-off stylistic nits unless they clearly recur
- If there are no patterns for a section, write: "None identified."
- Do NOT add any introductory or closing prose — output the four sections only
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_claude(comments_text: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    result = subprocess.run(
        ["/opt/homebrew/bin/claude", "-p", comments_text, "--system-prompt", _SYSTEM_PROMPT],
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited {result.returncode}: "
            f"stdout={result.stdout[:200]!r} stderr={result.stderr[:200]!r}"
        )
    return result.stdout.strip()


# ── Section parsing ────────────────────────────────────────────────────────────

_SECTION_TO_FILE = {
    "## Recurring Review Flags": "review-learnings.md",
    "## Team Anti-Patterns":     "anti-patterns.md",
    "## Architecture & Constraints": "architecture.md",
    "## Past Incident Patterns": "incidents.md",
}


def _parse_sections(analysis: str) -> dict[str, str]:
    headers = list(_SECTION_TO_FILE.keys())
    result: dict[str, str] = {}

    for i, header in enumerate(headers):
        start = analysis.find(header)
        if start == -1:
            logger.warning("Section %r not found in LLM output", header)
            continue
        # End of this section = start of the next one (or end of string)
        end = len(analysis)
        for other in headers[i + 1:]:
            pos = analysis.find(other, start + len(header))
            if pos != -1 and pos < end:
                end = pos
        content = analysis[start:end].strip()
        result[_SECTION_TO_FILE[header]] = content

    return result


# ── GitHub file writer ─────────────────────────────────────────────────────────

async def _get_existing_sha(
    client: httpx.AsyncClient, repo: str, path: str, headers: dict
) -> str | None:
    resp = await client.get(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=headers)
    return resp.json().get("sha") if resp.status_code == 200 else None


async def _write_file(
    client: httpx.AsyncClient,
    repo: str,
    filename: str,
    content: str,
    headers: dict,
    dry_run: bool,
    local_dir: str | None = None,
) -> None:
    path = f".mreview/{filename}"

    if local_dir:
        out = Path(local_dir) / filename
        out.write_text(content, encoding="utf-8")
        logger.info("Written locally: %s", out)
        return

    if dry_run:
        print(f"\n{'=' * 60}\n{path}\n{'=' * 60}\n{content}\n")
        return

    sha = await _get_existing_sha(client, repo, path, headers)
    body: dict = {
        "message": f"chore: update {path} from historical PR analysis",
        "content": base64.b64encode(content.encode()).decode(),
    }
    if sha:
        body["sha"] = sha

    resp = await client.put(
        f"{GITHUB_API}/repos/{repo}/contents/{path}",
        headers=headers,
        json=body,
    )
    resp.raise_for_status()
    logger.info("%s %s", "Updated" if sha else "Created", path)


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(repo: str, pr_count: int, dry_run: bool, local_dir: str | None = None) -> None:
    token = await get_installation_token(repo)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        logger.info("Fetching %d closed PRs from %s...", pr_count, repo)
        prs = await _fetch_closed_prs(client, repo, headers, pr_count)
        logger.info("Fetched %d PRs — loading review comments in parallel...", len(prs))

        inline_results, review_results = await asyncio.gather(
            asyncio.gather(*[_fetch_inline_comments(client, repo, pr["number"], headers) for pr in prs]),
            asyncio.gather(*[_fetch_review_bodies(client, repo, pr["number"], headers) for pr in prs]),
        )

        pr_data: list[dict] = []
        for pr, inline, reviews in zip(prs, inline_results, review_results):
            entry = _collect_substantive_comments(pr, inline, reviews)
            if entry:
                pr_data.append(entry)

        total_comments = sum(len(e["comments"]) for e in pr_data)
        logger.info(
            "Collected %d substantive comments from %d/%d PRs",
            total_comments, len(pr_data), len(prs),
        )

        if total_comments == 0:
            logger.warning("No substantive review comments found — nothing to analyze. Exiting.")
            return

        comments_text = _build_comments_text(pr_data)
        logger.info("Sending %d chars to Claude for analysis...", len(comments_text))
        analysis = _call_claude(comments_text)

        sections = _parse_sections(analysis)
        logger.info("Writing %d .mreview/ files%s", len(sections), " (dry run)" if dry_run else "")

        for filename, content in sections.items():
            await _write_file(client, repo, filename, content, headers, dry_run, local_dir)

    if local_dir:
        logger.info("Done — files written to %s", local_dir)
    elif dry_run:
        logger.info("Done — dry run, no files written to GitHub")
    else:
        logger.info("Done — .mreview/ files written to %s", repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate .mreview/ context files from historical PR review comments"
    )
    parser.add_argument("--repo", required=True, help="GitHub repo (e.g. Meesho/product-amplifyr)")
    parser.add_argument("--pr-count", type=int, default=50, help="Number of recent closed PRs to analyze (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Print output instead of writing to GitHub")
    parser.add_argument(
        "--local-dir",
        help="Write output files to this local directory instead of GitHub "
             "(default when omitting --local-dir: writes to GitHub; "
             "auto-detected path: mreview-context/{repo_name}/)",
    )
    args = parser.parse_args()

    asyncio.run(run(args.repo, args.pr_count, args.dry_run, args.local_dir))
