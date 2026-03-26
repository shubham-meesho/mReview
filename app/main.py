import hashlib
import hmac
import logging

from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from app.github.pr_fetcher import fetch_review_context
from app.models.webhook import WebhookPayload

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
        ctx = await fetch_review_context(payload)
        logger.info(
            "PR #%d — fetched %d reviewable files (lang: %s)",
            ctx.pr_number,
            len(ctx.diff_files),
            ctx.primary_language,
        )
        for f in ctx.diff_files:
            logger.debug("  [%s] %s (+%d/-%d)", f.status, f.filename, f.additions, f.deletions)
        # TODO: pass ctx to orchestrator once T7/T8 are done
    except Exception:
        logger.exception(
            "Failed to process PR #%d on %s",
            payload.pull_request.number,
            payload.repository.full_name,
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


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
