import hashlib
import hmac
import logging
import logging.config

from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import asyncio

from app.github.comment_poster import post_review
from app.github.context_fetcher import fetch_repo_context
from app.github.pr_fetcher import fetch_review_context
from app.models.webhook import WebhookPayload
from app.orchestrator import run_review

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(levelname)s  %(name)s — %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "app": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        "uvicorn.access": {"propagate": True},
    },
})

logger = logging.getLogger(__name__)

app = FastAPI(title="code-reviewer")

REVIEWED_ACTIONS = {"opened", "synchronize"}


def _verify_signature(body: bytes, secret: str, signature_header: str | None) -> None:
    """Raise 403 if the X-Hub-Signature-256 header doesn't match."""
    if not signature_header:
        raise HTTPException(status_code=403, detail="Missing X-Hub-Signature-256")
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=403, detail="Invalid signature")


async def _handle_pr(payload: WebhookPayload) -> None:
    """Background task: fetch PR diff, then hand off to the orchestrator."""
    logger.info(
        "Handling PR #%d (%s) on %s",
        payload.pull_request.number,
        payload.action,
        payload.repository.full_name,
    )
    try:
        # Fetch PR diff and .mreview/ context in parallel — both are independent GitHub API calls
        ctx, repo_context = await asyncio.gather(
            fetch_review_context(payload),
            fetch_repo_context(payload.repository.full_name),
        )
        ctx.repo_context = repo_context

        logger.info(
            "PR #%d — fetched %d reviewable files (lang: %s), repo context: %s",
            ctx.pr_number,
            len(ctx.diff_files),
            ctx.primary_language,
            repo_context.present_files() or "none",
        )
        # Store context so /review/inject can process manual Claude responses
        _pending_contexts[ctx.pr_number] = ctx
        comments = await run_review(ctx)
        logger.info("PR #%d — %d comments generated", ctx.pr_number, len(comments))
        for c in comments:
            logger.info("  [%s] %s:%d — %s", c.severity, c.file, c.line, c.message)
        await post_review(ctx, comments)
    except Exception:
        logger.exception(
            "Failed to process PR #%d on %s",
            payload.pull_request.number,
            payload.repository.full_name,
        )


# In-memory store of the last ReviewContext per PR (keyed by pr_number)
# Used by /review/inject to avoid re-fetching the diff
_pending_contexts: dict[int, object] = {}


class InjectRequest(BaseModel):
    pr_number: int
    claude_response: str  # raw JSON array pasted from Claude chat


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/review/inject")
async def inject_review(req: InjectRequest):
    """
    Manually inject a Claude response for a PR that was already fetched.

    Workflow:
      1. Fire /webhook — server fetches diff, writes prompt to prompts/pr_{n}.txt
      2. Paste that prompt into Claude chat, copy the JSON response
      3. POST here with pr_number + claude_response
      4. Server parses comments and logs them (T8 will post them to GitHub)
    """
    from app.agents.guidelines import _parse_comments

    ctx = _pending_contexts.get(req.pr_number)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pending context for PR #{req.pr_number}. Fire /webhook first.",
        )

    try:
        comments = _parse_comments(req.claude_response)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Claude response: {e}")

    logger.info("PR #%d — %d comments injected manually", req.pr_number, len(comments))
    for c in comments:
        logger.info("  [%s] %s:%d — %s", c.severity, c.file, c.line, c.message)

    await post_review(ctx, comments)

    return JSONResponse({
        "pr": req.pr_number,
        "comments_parsed": len(comments),
        "comments": [c.model_dump() for c in comments],
    })


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
):
    # 1. Read raw body first — signature must be verified against raw bytes
    body = await request.body()

    # 2. Lazy-import settings so tests can patch env vars before loading
    from app.config import settings

    _verify_signature(body, settings.github_webhook_secret, x_hub_signature_256)

    # 3. Only process pull_request events; ack everything else immediately
    if x_github_event != "pull_request":
        return JSONResponse({"ignored": True, "reason": "not a pull_request event"})

    payload = WebhookPayload.model_validate_json(body)

    # 4. Filter by action
    if payload.action not in REVIEWED_ACTIONS:
        return JSONResponse({"ignored": True, "reason": f"action={payload.action}"})

    # 5. Filter by target branch
    if payload.pull_request.base.ref != settings.target_branch:
        return JSONResponse({
            "ignored": True,
            "reason": f"base branch is {payload.pull_request.base.ref!r}, not {settings.target_branch!r}",
        })

    # 6. Enqueue review as a background task so GitHub gets a 200 immediately
    background_tasks.add_task(_handle_pr, payload)

    return JSONResponse({
        "accepted": True,
        "pr": payload.pull_request.number,
        "repo": payload.repository.full_name,
    })
