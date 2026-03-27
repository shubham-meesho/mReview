import base64
import logging
import uuid
from collections import Counter

import httpx
import pathspec

from app.github.app_auth import get_installation_token
from app.github.method_context import MAX_TOTAL_METHOD_LINES, extract_method_contexts
from app.models.review import FileDiff, ReviewContext
from app.models.webhook import WebhookPayload

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Hardcoded safety net — patterns that are never reviewable regardless of .gitignore
# Keep this list minimal; .gitignore is the primary source of truth.
ALWAYS_SKIP_PATTERNS = [
    "*_pb.go",           # Go protobuf generated
    "*.pb.java",         # Java protobuf generated
    "**/*.min.js",       # minified JS
    "**/*.min.css",      # minified CSS
    "**/*_generated*",   # generic generated marker
    "configs/**",        # infra config, not reviewable code
]

# Language inferred from file extension
_EXT_TO_LANG: dict[str, str] = {
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".go": "Go",
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".cs": "C#",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".swift": "Swift",
    ".scala": "Scala",
}

# Per-repo cache: repo_full_name → PathSpec (gitignore patterns)
_gitignore_cache: dict[str, pathspec.PathSpec] = {}


async def _fetch_gitignore_spec(
    client: httpx.AsyncClient,
    repo: str,
    token: str,
) -> pathspec.PathSpec:
    """
    Fetch the repo's root .gitignore and return a compiled PathSpec.
    Falls back to an empty spec if the file doesn't exist or can't be fetched.
    Results are cached per repo for the lifetime of the process.
    """
    if repo in _gitignore_cache:
        return _gitignore_cache[repo]

    try:
        response = await client.get(
            f"{GITHUB_API}/repos/{repo}/contents/.gitignore",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        if response.status_code == 404:
            logger.info("No .gitignore found in %s — skipping gitignore filtering", repo)
            spec = pathspec.PathSpec.from_lines("gitwildmatch", [])
        else:
            response.raise_for_status()
            content = base64.b64decode(response.json()["content"]).decode("utf-8")
            lines = content.splitlines()
            spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
            logger.info(
                "Loaded .gitignore from %s (%d pattern lines)", repo, len(lines)
            )
    except Exception as exc:
        logger.warning("Failed to fetch .gitignore for %s: %s — skipping", repo, exc)
        spec = pathspec.PathSpec.from_lines("gitwildmatch", [])

    _gitignore_cache[repo] = spec
    return spec


def _build_always_skip_spec() -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitwildmatch", ALWAYS_SKIP_PATTERNS)


_always_skip_spec = _build_always_skip_spec()


def _should_skip(filename: str, gitignore_spec: pathspec.PathSpec) -> bool:
    return (
        _always_skip_spec.match_file(filename)
        or gitignore_spec.match_file(filename)
    )


def _infer_language(files: list[FileDiff]) -> str:
    """Return the most common language across changed files, or 'Unknown'."""
    counts: Counter[str] = Counter()
    for f in files:
        ext = "." + f.filename.rsplit(".", 1)[-1] if "." in f.filename else ""
        lang = _EXT_TO_LANG.get(ext.lower())
        if lang:
            counts[lang] += 1
    return counts.most_common(1)[0][0] if counts else "Unknown"


async def fetch_review_context(payload: WebhookPayload) -> ReviewContext:
    """
    Fetch the changed files for a PR and return a ReviewContext ready
    to pass to an agent.

    Steps:
      1. Get an installation token via GitHub App auth
      2. Fetch .gitignore from the repo root (cached per repo)
      3. Call GET /repos/{owner}/{repo}/pulls/{number}/files
      4. Filter out: gitignored paths, always-skip patterns, binary/large files
      5. Infer the primary language from the remaining files
      6. Assemble and return a ReviewContext
    """
    repo = payload.repository.full_name
    pr_number = payload.pull_request.number

    token = await get_installation_token(repo)

    async with httpx.AsyncClient() as client:
        # Fetch .gitignore and PR files in parallel
        gitignore_spec, pr_files_response = await _fetch_in_parallel(
            client, repo, pr_number, token
        )

    pr_files_response.raise_for_status()
    raw_files: list[dict] = pr_files_response.json()
    all_count = len(raw_files)
    diff_files: list[FileDiff] = []

    for f in raw_files:
        filename = f["filename"]

        if _should_skip(filename, gitignore_spec):
            logger.debug("Skipping %s (gitignore or always-skip match)", filename)
            continue

        patch = f.get("patch")  # None for binary files or diffs too large for the API
        if patch is None:
            logger.debug("Skipping %s (binary or diff too large)", filename)
            continue

        diff_files.append(FileDiff(
            filename=filename,
            status=f["status"],
            additions=f["additions"],
            deletions=f["deletions"],
            patch=patch,
            blob_sha=f.get("sha"),
        ))

    primary_language = _infer_language(diff_files)
    logger.info(
        "PR #%d — %d files total, %d reviewable after filtering (lang: %s)",
        pr_number, all_count, len(diff_files), primary_language,
    )

    # Fetch enclosing method context for each changed file in parallel
    head_sha = payload.pull_request.head.sha or ""
    if head_sha:
        await _attach_method_contexts(diff_files, repo, head_sha, primary_language, token)
    else:
        logger.warning("PR #%d — no head SHA in payload, skipping method context", pr_number)

    return ReviewContext(
        run_id=str(uuid.uuid4()),
        repo_full_name=repo,
        pr_number=pr_number,
        pr_title=payload.pull_request.title,
        pr_description=payload.pull_request.body or "",
        primary_language=primary_language,
        diff_files=diff_files,
    )


async def _attach_method_contexts(
    diff_files: list[FileDiff],
    repo: str,
    head_sha: str,
    lang: str,
    token: str,
) -> None:
    """
    Fetch enclosing method source for every changed file in parallel.
    Respects MAX_TOTAL_METHOD_LINES — stops attaching once the budget is used.
    Failures are logged and ignored so a blob fetch error never aborts the review.
    """
    import asyncio

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[extract_method_contexts(client, repo, head_sha, headers, f, lang) for f in diff_files],
            return_exceptions=True,
        )

    total_lines = 0
    for f, result in zip(diff_files, results):
        if isinstance(result, Exception):
            logger.warning("Method context extraction failed for %s: %s", f.filename, result)
            continue
        for ctx in result:
            ctx_lines = ctx.source.count("\n") + 1
            if total_lines + ctx_lines > MAX_TOTAL_METHOD_LINES:
                logger.info("Method context budget exhausted — skipping remaining files")
                return
            f.method_contexts.append(ctx)
            total_lines += ctx_lines


async def _fetch_in_parallel(
    client: httpx.AsyncClient,
    repo: str,
    pr_number: int,
    token: str,
) -> tuple[pathspec.PathSpec, httpx.Response]:
    """Fire the .gitignore fetch and PR files fetch concurrently."""
    import asyncio

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    gitignore_task = asyncio.create_task(
        _fetch_gitignore_spec(client, repo, token)
    )
    pr_files_task = asyncio.create_task(
        client.get(
            f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
            headers=headers,
            params={"per_page": 100},
        )
    )

    gitignore_spec, pr_files_response = await asyncio.gather(
        gitignore_task, pr_files_task
    )
    return gitignore_spec, pr_files_response
