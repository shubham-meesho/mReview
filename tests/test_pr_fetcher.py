import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pathspec
import pytest

from app.github.pr_fetcher import (
    _gitignore_cache,
    _infer_language,
    _should_skip,
    fetch_review_context,
)
from app.models.review import FileDiff
from app.models.webhook import PullRequest, PullRequestRef, Repository, WebhookPayload


# ── helpers ──────────────────────────────────────────────────────────────────

def _empty_spec() -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitwildmatch", [])


def _spec(*patterns: str) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitwildmatch", list(patterns))


def _make_payload() -> WebhookPayload:
    return WebhookPayload(
        action="opened",
        pull_request=PullRequest(
            number=7,
            title="My PR",
            body="desc",
            head=PullRequestRef(ref="feature/x"),
            base=PullRequestRef(ref="develop"),
        ),
        repository=Repository(full_name="org/repo"),
    )


def _file(filename: str, patch: str | None = "@@ +line") -> dict:
    return {
        "filename": filename,
        "status": "modified",
        "additions": 1,
        "deletions": 0,
        "patch": patch,
    }


def _mock_http(raw_files: list[dict], gitignore_content: str = "") -> tuple:
    """Return (mock_client_cls, gitignore_response, pr_files_response)."""
    # gitignore response
    gi_resp = MagicMock()
    gi_resp.status_code = 200
    gi_resp.raise_for_status = MagicMock()
    gi_resp.json.return_value = {
        "content": base64.b64encode(gitignore_content.encode()).decode()
    }

    # PR files response
    pr_resp = MagicMock()
    pr_resp.raise_for_status = MagicMock()
    pr_resp.json.return_value = raw_files

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[gi_resp, pr_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_client)
    return mock_cls, gi_resp, pr_resp


# ── _should_skip (always-skip patterns) ──────────────────────────────────────

def test_always_skip_protobuf_go():
    assert _should_skip("proto/foo_pb.go", _empty_spec()) is True

def test_always_skip_minified_js():
    assert _should_skip("static/bundle.min.js", _empty_spec()) is True

def test_always_skip_configs_dir():
    assert _should_skip("configs/service/application-prd.yml", _empty_spec()) is True

def test_no_skip_normal_java():
    assert _should_skip("src/main/java/Foo.java", _empty_spec()) is False

def test_no_skip_normal_go():
    assert _should_skip("internal/handler.go", _empty_spec()) is False


# ── _should_skip (gitignore patterns) ────────────────────────────────────────

def test_gitignore_vendor_skipped():
    spec = _spec("vendor/")
    assert _should_skip("vendor/lib/util.go", spec) is True

def test_gitignore_target_dir_skipped():
    spec = _spec("target/")
    assert _should_skip("target/classes/Foo.class", spec) is True

def test_gitignore_does_not_affect_normal_files():
    spec = _spec("target/", "*.class")
    assert _should_skip("src/main/java/Foo.java", spec) is False

def test_gitignore_negation_keeps_file():
    # !important.log should NOT be skipped even if *.log is ignored
    spec = _spec("*.log", "!important.log")
    assert _should_skip("important.log", spec) is False


# ── _infer_language ───────────────────────────────────────────────────────────

def test_infer_java():
    files = [FileDiff("Foo.java", "modified", 1, 0, ""), FileDiff("Bar.java", "added", 1, 0, "")]
    assert _infer_language(files) == "Java"

def test_infer_majority_wins():
    files = [
        FileDiff("A.java", "modified", 1, 0, ""),
        FileDiff("B.java", "modified", 1, 0, ""),
        FileDiff("C.go", "modified", 1, 0, ""),
    ]
    assert _infer_language(files) == "Java"

def test_infer_unknown_for_no_known_extensions():
    assert _infer_language([FileDiff("README.md", "modified", 1, 0, "")]) == "Unknown"

def test_infer_empty_list():
    assert _infer_language([]) == "Unknown"


# ── fetch_review_context ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_builds_correct_context():
    _gitignore_cache.clear()
    mock_cls, _, _ = _mock_http(
        raw_files=[_file("src/Foo.java"), _file("src/Bar.java")],
        gitignore_content="target/\n*.class\n",
    )
    with (
        patch("app.github.pr_fetcher.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        ctx = await fetch_review_context(_make_payload())

    assert ctx.repo_full_name == "org/repo"
    assert ctx.pr_number == 7
    assert ctx.primary_language == "Java"
    assert len(ctx.diff_files) == 2
    assert ctx.run_id


@pytest.mark.asyncio
async def test_gitignore_patterns_filter_files():
    _gitignore_cache.clear()
    mock_cls, _, _ = _mock_http(
        raw_files=[
            _file("src/Foo.java"),
            _file("vendor/lib/util.go"),   # gitignored
            _file("target/Foo.class"),     # gitignored
        ],
        gitignore_content="vendor/\ntarget/\n",
    )
    with (
        patch("app.github.pr_fetcher.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        ctx = await fetch_review_context(_make_payload())

    assert len(ctx.diff_files) == 1
    assert ctx.diff_files[0].filename == "src/Foo.java"


@pytest.mark.asyncio
async def test_configs_always_skipped():
    _gitignore_cache.clear()
    mock_cls, _, _ = _mock_http(
        raw_files=[
            _file("src/Foo.java"),
            _file("configs/service/application-prd.yml"),  # always-skip
        ],
        gitignore_content="",
    )
    with (
        patch("app.github.pr_fetcher.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        ctx = await fetch_review_context(_make_payload())

    assert len(ctx.diff_files) == 1
    assert ctx.diff_files[0].filename == "src/Foo.java"


@pytest.mark.asyncio
async def test_binary_files_filtered_out():
    _gitignore_cache.clear()
    mock_cls, _, _ = _mock_http(
        raw_files=[_file("src/Foo.java"), _file("assets/img.png", patch=None)],
        gitignore_content="",
    )
    with (
        patch("app.github.pr_fetcher.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        ctx = await fetch_review_context(_make_payload())

    assert len(ctx.diff_files) == 1
    assert ctx.diff_files[0].filename == "src/Foo.java"


@pytest.mark.asyncio
async def test_missing_gitignore_falls_back_gracefully():
    _gitignore_cache.clear()

    gi_resp = MagicMock()
    gi_resp.status_code = 404

    pr_resp = MagicMock()
    pr_resp.raise_for_status = MagicMock()
    pr_resp.json.return_value = [_file("src/Foo.java")]

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[gi_resp, pr_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.github.pr_fetcher.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", MagicMock(return_value=mock_client)),
    ):
        ctx = await fetch_review_context(_make_payload())

    assert len(ctx.diff_files) == 1


@pytest.mark.asyncio
async def test_gitignore_is_cached_on_second_call():
    _gitignore_cache.clear()

    # Pre-seed the cache — gitignore should NOT be fetched
    _gitignore_cache["org/repo"] = _spec("target/")

    # Only one response needed: the PR files call (gitignore is skipped)
    pr_resp = MagicMock()
    pr_resp.raise_for_status = MagicMock()
    pr_resp.json.return_value = [_file("src/Foo.java")]

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=pr_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_cls = MagicMock(return_value=mock_client)

    with (
        patch("app.github.pr_fetcher.get_installation_token", new=AsyncMock(return_value="tok")),
        patch("httpx.AsyncClient", mock_cls),
    ):
        ctx = await fetch_review_context(_make_payload())

    # Only 1 HTTP call (PR files) — gitignore was served from cache
    assert mock_client.get.call_count == 1
    assert len(ctx.diff_files) == 1
