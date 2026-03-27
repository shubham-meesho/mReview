import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.guidelines import (
    CONFIDENCE_THRESHOLD,
    GuidelinesAgent,
    _annotate_patch,
    _build_diff_content,
    _parse_comments,
)
from app.models.review import FileDiff, ReviewContext


# ── helpers ──────────────────────────────────────────────────────────────────

def _ctx(files: list[FileDiff], lang: str = "Java") -> ReviewContext:
    return ReviewContext(
        run_id="test-run",
        repo_full_name="org/repo",
        pr_number=42,
        pr_title="Test PR",
        pr_description="",
        primary_language=lang,
        diff_files=files,
    )


def _file(name: str = "src/Foo.java", patch: str = "@@ -1 +1 @@\n+new line") -> FileDiff:
    return FileDiff(filename=name, status="modified", additions=1, deletions=0, patch=patch)


def _comment(**kwargs) -> dict:
    base = {
        "file": "src/Foo.java",
        "line": 10,
        "severity": "WARNING",
        "category": "error-handling",
        "message": "Exception swallowed silently.",
        "rationale": "Silent exceptions hide bugs.",
        "confidence": 0.9,
    }
    return {**base, **kwargs}


def _claude_response(comments: list[dict]) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(comments))]
    return msg


# ── _annotate_patch ───────────────────────────────────────────────────────────

def test_annotate_patch_context_lines_get_line_numbers():
    patch = "@@ -10,3 +10,3 @@\n context\n+added\n context"
    annotated, lines = _annotate_patch(patch)
    assert "L0010:  context" in annotated
    assert "L0011: +added" in annotated
    assert "L0012:  context" in annotated
    assert lines == 3


def test_annotate_patch_deleted_lines_get_dashes():
    patch = "@@ -1,2 +1,1 @@\n context\n-removed"
    annotated, _ = _annotate_patch(patch)
    assert "L----: -removed" in annotated


def test_annotate_patch_line_count_excludes_deleted():
    patch = "@@ -1,3 +1,2 @@\n context\n-removed\n context"
    _, lines = _annotate_patch(patch)
    assert lines == 2  # context + context, not the deleted line


def test_annotate_patch_multiple_hunks():
    patch = "@@ -1,1 +1,1 @@\n+line1\n@@ -10,1 +10,1 @@\n+line10"
    annotated, lines = _annotate_patch(patch)
    assert "L0001: +line1" in annotated
    assert "L0010: +line10" in annotated
    assert lines == 2


def test_annotate_patch_empty():
    annotated, lines = _annotate_patch("")
    assert annotated == ""
    assert lines == 0


# ── _build_diff_content ───────────────────────────────────────────────────────

def test_build_diff_includes_all_files():
    files = [_file("A.java", "@@ +line1"), _file("B.java", "@@ +line2")]
    content, _ = _build_diff_content(files)
    assert "A.java" in content
    assert "B.java" in content


def test_build_diff_truncates_at_max_lines():
    from app.agents.guidelines import MAX_DIFF_LINES
    big_patch = "\n".join(["+line"] * (MAX_DIFF_LINES + 500))
    files = [_file("Big.java", big_patch)]
    content, lines = _build_diff_content(files)
    assert lines == MAX_DIFF_LINES
    assert "truncated" in content


def test_build_diff_skips_none_patch():
    files = [FileDiff("bin.png", "added", 0, 0, None)]
    content, lines = _build_diff_content(files)
    assert content == ""
    assert lines == 0


# ── _parse_comments ───────────────────────────────────────────────────────────

def test_parse_valid_json():
    raw = json.dumps([_comment()])
    comments = _parse_comments(raw)
    assert len(comments) == 1
    assert comments[0].severity == "WARNING"


def test_parse_strips_markdown_fences():
    raw = "```json\n" + json.dumps([_comment()]) + "\n```"
    comments = _parse_comments(raw)
    assert len(comments) == 1


def test_parse_filters_low_confidence():
    raw = json.dumps([
        _comment(confidence=0.9),
        _comment(confidence=0.3),   # below threshold
        _comment(confidence=CONFIDENCE_THRESHOLD),  # exactly at threshold — keep
    ])
    comments = _parse_comments(raw)
    assert len(comments) == 2


def test_parse_skips_invalid_objects_gracefully():
    raw = json.dumps([_comment(), {"bad": "object"}])
    comments = _parse_comments(raw)
    assert len(comments) == 1


def test_parse_raises_on_non_array():
    with pytest.raises(ValueError):
        _parse_comments(json.dumps({"not": "an array"}))


def test_parse_raises_on_invalid_json():
    with pytest.raises(Exception):
        _parse_comments("not json at all")


# ── GuidelinesAgent.review ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_review_returns_comments_from_claude():
    agent = GuidelinesAgent()
    agent._backend = "api"
    ctx = _ctx([_file()])
    expected = [_comment()]

    with patch.object(agent._client.messages, "create", return_value=_claude_response(expected)):
        comments = await agent.review(ctx)

    assert len(comments) == 1
    assert comments[0].file == "src/Foo.java"


@pytest.mark.asyncio
async def test_review_empty_files_returns_no_comments():
    agent = GuidelinesAgent()
    ctx = _ctx([])

    comments = await agent.review(ctx)

    assert comments == []


@pytest.mark.asyncio
async def test_review_retries_on_bad_json():
    agent = GuidelinesAgent()
    agent._backend = "api"
    ctx = _ctx([_file()])

    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="this is not json")]

    good_response = _claude_response([_comment()])

    with patch.object(
        agent._client.messages, "create",
        side_effect=[bad_response, good_response]
    ):
        comments = await agent.review(ctx)

    assert len(comments) == 1


@pytest.mark.asyncio
async def test_review_returns_empty_if_both_attempts_fail():
    agent = GuidelinesAgent()
    agent._backend = "api"
    ctx = _ctx([_file()])

    bad = MagicMock()
    bad.content = [MagicMock(text="not json")]

    with patch.object(agent._client.messages, "create", return_value=bad):
        comments = await agent.review(ctx)

    assert comments == []


@pytest.mark.asyncio
async def test_review_uses_language_specific_rules():
    agent = GuidelinesAgent()
    agent._backend = "api"
    ctx = _ctx([_file()], lang="Go")

    captured_system: list[str] = []

    def capture(**kwargs):
        captured_system.append(kwargs.get("system", ""))
        return _claude_response([])

    with patch.object(agent._client.messages, "create", side_effect=capture):
        await agent.review(ctx)

    assert "context.Context" in captured_system[0]
    assert "Language: Go" in captured_system[0]
