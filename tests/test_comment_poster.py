import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.github.comment_poster import (
    _build_summary,
    _format_inline_body,
    _lines_in_diff,
    _reviewable_lines,
    post_review,
)
from app.models.review import Comment, FileDiff, ReviewContext


# ── helpers ──────────────────────────────────────────────────────────────────

def _ctx(files: list[FileDiff] | None = None) -> ReviewContext:
    return ReviewContext(
        run_id="run-1",
        repo_full_name="org/repo",
        pr_number=42,
        pr_title="My PR",
        pr_description="",
        primary_language="Java",
        diff_files=files or [],
    )


def _comment(**kwargs) -> Comment:
    base = dict(
        file="src/Foo.java",
        line=10,
        severity="WARNING",
        category="error-handling",
        message="Exception swallowed.",
        rationale="Hides bugs.",
        confidence=0.9,
    )
    return Comment(**{**base, **kwargs})


def _mock_http(head_sha: str = "abc1234") -> tuple[MagicMock, MagicMock, MagicMock]:
    pr_resp = MagicMock()
    pr_resp.raise_for_status = MagicMock()
    pr_resp.json.return_value = {"head": {"sha": head_sha}}

    review_resp = MagicMock()
    review_resp.raise_for_status = MagicMock()
    review_resp.json.return_value = {"id": 99}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=pr_resp)
    mock_client.post = AsyncMock(return_value=review_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=mock_client), pr_resp, review_resp


# ── _lines_in_diff ────────────────────────────────────────────────────────────

def test_lines_in_diff_basic():
    patch = "@@ -1,3 +1,4 @@\n context\n+added line\n context\n context"
    lines = _lines_in_diff(patch)
    assert 2 in lines   # added line is at new-file line 2
    assert 1 in lines   # context line at line 1
    assert 3 in lines


def test_lines_in_diff_deleted_lines_not_included():
    patch = "@@ -1,3 +1,2 @@\n context\n-deleted line\n context"
    lines = _lines_in_diff(patch)
    assert 2 in lines   # context after deleted
    # deleted line has no new-file number
    assert len(lines) == 2


def test_lines_in_diff_multiple_hunks():
    patch = (
        "@@ -1,2 +1,2 @@\n context\n+line2\n"
        "@@ -10,2 +10,2 @@\n context\n+line11\n"
    )
    lines = _lines_in_diff(patch)
    assert 2 in lines
    assert 11 in lines


def test_lines_in_diff_empty_patch():
    assert _lines_in_diff("") == set()


# ── _reviewable_lines ─────────────────────────────────────────────────────────

def test_reviewable_lines_maps_filename_to_lines():
    files = [FileDiff("src/Foo.java", "modified", 1, 0, "@@ -1 +1 @@\n+new")]
    result = _reviewable_lines(_ctx(files))
    assert "src/Foo.java" in result
    assert 1 in result["src/Foo.java"]


def test_reviewable_lines_skips_none_patch():
    files = [FileDiff("img.png", "added", 0, 0, None)]
    result = _reviewable_lines(_ctx(files))
    assert "img.png" not in result


# ── _format_inline_body ───────────────────────────────────────────────────────

def test_format_inline_body_contains_severity_and_message():
    c = _comment(severity="BLOCKER", message="Use try-with-resources.")
    body = _format_inline_body(c)
    assert "BLOCKER" in body
    assert "Use try-with-resources." in body
    assert "🚨" in body


# ── _build_summary ────────────────────────────────────────────────────────────

def test_build_summary_counts_severities():
    comments = [
        _comment(severity="BLOCKER"),
        _comment(severity="WARNING"),
        _comment(severity="WARNING"),
    ]
    summary = _build_summary(comments, _ctx())
    assert "BLOCKER" in summary
    assert "WARNING" in summary
    assert "PR #42" in summary


def test_build_summary_no_comments():
    summary = _build_summary([], _ctx())
    assert "No issues found" in summary


# ── post_review ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_post_review_sends_inline_comments():
    diff = "@@ -1,5 +1,6 @@\n context\n+line2\n context\n+line4\n context\n+line6"
    files = [FileDiff("src/Foo.java", "modified", 3, 0, diff)]
    ctx = _ctx(files)
    comments = [_comment(line=2), _comment(line=4)]

    mock_cls, _, review_resp = _mock_http()

    with (
        patch("app.github.comment_poster.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        await post_review(ctx, comments)

    # Check the review POST was called with both inline comments
    call_kwargs = mock_cls.return_value.post.call_args_list[0][1]
    payload = call_kwargs["json"]
    assert len(payload["comments"]) == 2
    assert payload["event"] == "COMMENT"
    assert payload["commit_id"] == "abc1234"


@pytest.mark.asyncio
async def test_post_review_fallback_for_invalid_lines():
    diff = "@@ -1,2 +1,2 @@\n context\n+line2"
    files = [FileDiff("src/Foo.java", "modified", 1, 0, diff)]
    ctx = _ctx(files)
    # line 99 is NOT in the diff — should fall back to issue comment
    comments = [_comment(line=99)]

    mock_cls, _, _ = _mock_http()

    with (
        patch("app.github.comment_poster.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        await post_review(ctx, comments)

    posts = mock_cls.return_value.post.call_args_list
    # First POST: review (with 0 inline comments)
    review_payload = posts[0][1]["json"]
    assert review_payload["comments"] == []
    # Second POST: fallback issue comment
    assert len(posts) == 2
    fallback_url = posts[1][0][0]
    assert "issues" in fallback_url


@pytest.mark.asyncio
async def test_post_review_empty_comments_still_posts_summary():
    ctx = _ctx([FileDiff("src/Foo.java", "modified", 1, 0, "@@ -1 +1 @@\n+line")])
    mock_cls, _, _ = _mock_http()

    with (
        patch("app.github.comment_poster.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        await post_review(ctx, [])

    review_payload = mock_cls.return_value.post.call_args_list[0][1]["json"]
    assert "No issues found" in review_payload["body"]
    assert review_payload["comments"] == []
