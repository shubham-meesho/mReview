import logging
import re

import httpx

from app.github.app_auth import get_installation_token
from app.models.review import Comment, ReviewContext

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

_SEVERITY_EMOJI = {
    "BLOCKER":    "🚨",
    "WARNING":    "⚠️",
    "SUGGESTION": "💡",
    "NITPICK":    "🔧",
}


def _lines_in_diff(patch: str) -> set[int]:
    """
    Return the set of new-file line numbers that appear in this diff hunk.
    Only lines starting with '+' (added/context) are valid targets for
    GitHub inline review comments using side=RIGHT.
    """
    lines: set[int] = set()
    current_line = 0
    for raw in patch.splitlines():
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if hunk:
            current_line = int(hunk.group(1)) - 1
            continue
        if raw.startswith("-"):
            continue  # deleted line — no new-file line number
        current_line += 1
        lines.add(current_line)
    return lines


def _reviewable_lines(context: ReviewContext) -> dict[str, set[int]]:
    """Map filename → set of new-file line numbers present in the diff."""
    return {
        f.filename: _lines_in_diff(f.patch)
        for f in context.diff_files
        if f.patch
    }


def _format_inline_body(comment: Comment) -> str:
    emoji = _SEVERITY_EMOJI.get(comment.severity, "💬")
    return (
        f"{emoji} **{comment.severity}** — {comment.message}\n\n"
        f"{comment.rationale}"
    )


def _build_summary(comments: list[Comment], context: ReviewContext) -> str:
    counts: dict[str, int] = {}
    for c in comments:
        counts[c.severity] = counts.get(c.severity, 0) + 1

    rows = "\n".join(
        f"| {sev} | {_SEVERITY_EMOJI.get(sev, '')} | {cnt} |"
        for sev, cnt in sorted(counts.items())
    )
    table = (
        "| Severity | | Count |\n"
        "|----------|---|-------|\n"
        f"{rows}"
    ) if rows else "_No issues found._"

    return (
        f"## 🤖 Code Review — PR #{context.pr_number}\n\n"
        f"**Language:** {context.primary_language} &nbsp;|&nbsp; "
        f"**Files reviewed:** {len(context.diff_files)}\n\n"
        f"{table}\n\n"
        f"_Reviewed by GuidelinesAgent · `claude-sonnet-4-6`_"
    )


async def post_review(context: ReviewContext, comments: list[Comment]) -> None:
    """
    Post a GitHub PR Review containing:
      - Inline comments for each Comment whose line exists in the diff
      - A PR-level summary comment as the review body
      - Fallback general comments for any lines not in the diff
    """
    repo = context.repo_full_name
    pr_number = context.pr_number

    token = await get_installation_token(repo)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient() as client:
        # Fetch the latest commit SHA — required for submitting a review
        head_sha = await _get_head_sha(client, repo, pr_number, headers)

        reviewable = _reviewable_lines(context)
        inline: list[dict] = []
        fallback: list[Comment] = []

        for c in comments:
            valid_lines = reviewable.get(c.file, set())
            if c.line in valid_lines:
                inline.append({
                    "path": c.file,
                    "line": c.line,
                    "side": "RIGHT",
                    "body": _format_inline_body(c),
                })
            else:
                logger.warning(
                    "Line %d not in diff for %s — will post as general comment",
                    c.line, c.file,
                )
                fallback.append(c)

        summary = _build_summary(comments, context)

        # Submit the review with all inline comments + summary body
        review_resp = await client.post(
            f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews",
            headers=headers,
            json={
                "commit_id": head_sha,
                "event": "COMMENT",
                "body": summary,
                "comments": inline,
            },
        )
        review_resp.raise_for_status()
        logger.info(
            "PR #%d — review posted with %d inline comments (commit %s)",
            pr_number, len(inline), head_sha[:7],
        )

        # Post fallback comments as regular PR issue comments
        for c in fallback:
            body = (
                f"**{_SEVERITY_EMOJI.get(c.severity, '')} {c.severity}** "
                f"in `{c.file}` (line {c.line})\n\n"
                f"{c.message}\n\n{c.rationale}"
            )
            fb_resp = await client.post(
                f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
                headers=headers,
                json={"body": body},
            )
            fb_resp.raise_for_status()
            logger.info("PR #%d — fallback comment posted for %s:%d", pr_number, c.file, c.line)


async def _get_head_sha(
    client: httpx.AsyncClient,
    repo: str,
    pr_number: int,
    headers: dict,
) -> str:
    resp = await client.get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}",
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()["head"]["sha"]
