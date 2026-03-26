"""
Quick smoke-test for T6: fetch a real PR diff using a PAT.

Usage:
    GITHUB_TOKEN=ghp_xxx python scripts/test_fetch.py <pr_url>

Example:
    GITHUB_TOKEN=ghp_xxx python scripts/test_fetch.py https://github.com/Meesho/store-front/pull/3756
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimal env so config doesn't complain
os.environ.setdefault("GITHUB_APP_ID", "placeholder")
os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "placeholder")
os.environ.setdefault("ANTHROPIC_API_KEY", "placeholder")
os.environ.setdefault("TARGET_BRANCH", "develop")

from app.github.pr_fetcher import fetch_review_context
from app.models.webhook import PullRequest, PullRequestRef, Repository, WebhookPayload


def parse_pr_url(url: str) -> tuple[str, int]:
    """Parse 'https://github.com/{owner}/{repo}/pull/{number}' → (full_name, number)."""
    parts = url.rstrip("/").split("/")
    if len(parts) < 7 or parts[5] != "pull":
        raise ValueError(f"Unexpected PR URL format: {url}")
    return f"{parts[3]}/{parts[4]}", int(parts[6])


async def main(pr_url: str, token: str) -> None:
    repo_full_name, pr_number = parse_pr_url(pr_url)
    print(f"Fetching PR #{pr_number} from {repo_full_name} …\n")

    payload = WebhookPayload(
        action="opened",
        pull_request=PullRequest(
            number=pr_number,
            title="(test)",
            body="",
            head=PullRequestRef(ref="feature"),
            base=PullRequestRef(ref="develop"),
        ),
        repository=Repository(full_name=repo_full_name),
    )

    # Bypass GitHub App auth — inject the PAT directly
    with patch(
        "app.github.pr_fetcher.get_installation_token",
        new=AsyncMock(return_value=token),
    ):
        ctx = await fetch_review_context(payload)

    print(f"run_id          : {ctx.run_id}")
    print(f"repo            : {ctx.repo_full_name}")
    print(f"pr_number       : {ctx.pr_number}")
    print(f"primary_language: {ctx.primary_language}")
    print(f"reviewable files: {len(ctx.diff_files)}\n")

    for f in ctx.diff_files:
        print(f"  [{f.status:8s}] {f.filename}  (+{f.additions}/-{f.deletions})")

    if not ctx.diff_files:
        print("  (no reviewable files after filtering)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: set GITHUB_TOKEN env var to a PAT with 'repo' scope.")
        sys.exit(1)

    asyncio.run(main(sys.argv[1], github_token))
