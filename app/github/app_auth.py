import logging
import time
from dataclasses import dataclass

import httpx
import jwt

from app.config import settings

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
# Installation tokens expire in 1h; refresh when less than this many seconds remain
_REFRESH_BUFFER_SECS = 300  # 5 minutes


@dataclass
class _CachedToken:
    token: str
    expires_at: float  # unix timestamp


# Per-installation token cache: installation_id → CachedToken
_token_cache: dict[int, _CachedToken] = {}


def _build_jwt(app_id: str, private_key_pem: str) -> str:
    """
    Generate a GitHub App JWT valid for ~9 minutes.

    iat is set 60s in the past to absorb clock skew between this machine
    and GitHub's servers — without this, GitHub occasionally rejects tokens
    whose iat is slightly in GitHub's future.
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (9 * 60),
        "iss": app_id,
    }
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


def _load_private_key(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


async def _get_installation_id(app_jwt: str, repo_full_name: str) -> int:
    """
    Find the installation ID for a given repo.
    Calls GET /repos/{owner}/{repo}/installation.
    """
    owner, repo = repo_full_name.split("/", 1)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/installation",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        response.raise_for_status()
        return response.json()["id"]


async def _fetch_installation_token(app_jwt: str, installation_id: int) -> _CachedToken:
    """Exchange a JWT for a scoped installation access token."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        response.raise_for_status()
        data = response.json()

    # expires_at is ISO-8601, e.g. "2024-01-01T01:00:00Z"
    from datetime import datetime, timezone
    expires_at = datetime.fromisoformat(
        data["expires_at"].replace("Z", "+00:00")
    ).timestamp()

    return _CachedToken(token=data["token"], expires_at=expires_at)


async def get_installation_token(repo_full_name: str) -> str:
    """
    Return a valid installation access token for the given repo.

    Handles the full two-step GitHub App auth flow:
      1. Build a signed JWT from the App's private key
      2. Exchange it for a per-installation token

    Tokens are cached and transparently refreshed when less than
    5 minutes of validity remain.
    """
    private_key = _load_private_key(settings.github_app_private_key_path)
    app_jwt = _build_jwt(settings.github_app_id, private_key)

    installation_id = await _get_installation_id(app_jwt, repo_full_name)

    cached = _token_cache.get(installation_id)
    if cached and time.time() < cached.expires_at - _REFRESH_BUFFER_SECS:
        logger.debug("Using cached token for installation %d", installation_id)
        return cached.token

    logger.info("Fetching new installation token for installation %d", installation_id)
    cached = await _fetch_installation_token(app_jwt, installation_id)
    _token_cache[installation_id] = cached
    return cached.token
