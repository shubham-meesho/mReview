import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.github.app_auth import (
    _build_jwt,
    _token_cache,
    get_installation_token,
)

# A real 2048-bit RSA key generated for testing only — not used for real auth
TEST_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1iAFQMBRU9/pLE51xQ6Mz2sJouc32HRXdQW3NYVjVXGOIebm
F04lC94Merl9cIel8PWj4FPuuTqXbgJB3QYQby9/T1q/DnKgnJtq66hcg8ISAPeT
4W7VkmPFdsdmn4JjnAvRcXhw74a3DSn2dLuMxEIdjG3m50IRpDw1zEOsAv406yM4
CMFPKq8x4SdruaEi37Z9TJ8n2x8CqcYK2aN7MnC7aeIDWIs/UfYzC7dZjZp0wNcr
V+CUj1nhx+Kf8AKsOzGsnF6bXGMVmNr4f0pd8ozBCl2SjEN7FRa8XMhGeyU2ixRv
5mg19KGXhig3ji99YQrgo55R9RdnZFDfiRyT5wIDAQABAoIBAFlEfCwkgUNQY+te
wmZmzHlkjF3nuzZ4OkXFHj4n3+OtNa4YjvBEWdl6twUq84rIYnv0TF+nXizGqn7o
XgEBGDTiPlcc4+3oB+GCQa8BP6CGde8FSBeBl3QyOA5uzu/M8i4KF0URCzQwm3vX
dLHxxpvDXIJBBzR+LNClcWA5Disof9D6lHSTaCOuIU9asPsYzdZFuERXv6z0gn4L
rKmN4yQBS2GlxyZIC+5DWSxudIDxmbKdHhkGsEGp+LwpaENw03q4N1BjpblOtvga
/zggpo3/PTPg7hn8x3N2cxKkNFL6mcfCR/lZrJc6BE8eJOFsY4SIrwp+pdQ4opLo
rKaEPKECgYEA+fQUB1jZr4XCHoGqLs6LfYGa4n7/FSJLZSQ3ujDUjEN0/ZNkATx+
8/YbjVdqn013MgJ/to3sTdmDULxsqccObYnXuIelXZUGFPYfYf4rio2l75c0HOpm
SXBjUy8VAzx8PaMVBZ60PTHGHtj0FPW1JkZNOAL4j4/8y518oN3B5jkCgYEA204Q
Ieja979ApPvLkNq6HRWD51DgXjxIRhP+PUVyTC6jJaVPGIpH+kUq1rqUt0ICveDt
U+/uh6h0viZBxKRv4Ryxp2Sx1pJOxtETaE0QOKf9q0kqYL920057Pl4JP8RR8GUo
edGNzMo3WvuMleKLDAtQ/hGJenvBcElgFTmnSx8CgYEA9Knb6Q0pl8vgJahagxKe
K63vhAE3guJc1pXLDq+5fcqR+5yIaUVkJz0h3wpQTbqwD/2uMW1efx2IkkC+RLmZ
/9LLm877KU0z9NSuB2eaCzd307w3wI4YrY4bS1NsyAwuuM6nRyb/2l6cRZmr4dBZ
DywFbexKjYwtsMlpMrWelNECgYAJFBU04ifWY7NwYQAPUg6sT4yzUbTIEeTICeHX
dX2Cy9dzIoHAuVC7eouIchbf8zqb06jfdapBMWcLzBei5U/AWOi9zjxSbqIWNud/
xNjsD4A/y/qWILbxjmkprsjhO+ZMdlOTn2ooVSKvgIRUXBl3eSx0KBOx31elp1Pz
7PwQ7wKBgQCHQHIBdMxAFj2qYg+/hsHsuQbf8q9xTxygHffEjFrnwG9A32XDtq7x
4/fzi52lSDMze+275On5nXujMAcZKA9n8q8InmrEhCJrNOXKcIh6/PW1qjBuYSYt
/oUvsmZdjgFJIo4w/7egLIjp9aML0iowi6Lm9Vb2PygphmcEOacIhg==
-----END RSA PRIVATE KEY-----"""


# --- JWT ---

def test_build_jwt_contains_expected_claims():
    import jwt as pyjwt
    token = _build_jwt("123456", TEST_PRIVATE_KEY)
    # Decode without verification to inspect claims
    claims = pyjwt.decode(token, options={"verify_signature": False})
    assert claims["iss"] == "123456"
    now = int(time.time())
    # iat is 60s in the past
    assert claims["iat"] <= now - 59
    # exp is ~9 minutes from now
    assert now + 8 * 60 <= claims["exp"] <= now + 10 * 60


def test_build_jwt_uses_rs256():
    import jwt as pyjwt
    token = _build_jwt("123456", TEST_PRIVATE_KEY)
    header = pyjwt.get_unverified_header(token)
    assert header["alg"] == "RS256"


# --- get_installation_token ---

def _mock_settings():
    """Return a MagicMock that looks like app.config.settings."""
    s = MagicMock()
    s.github_app_id = "123456"
    s.github_app_private_key_path = "./keys/github-app.pem"
    return s


@pytest.mark.asyncio
async def test_get_installation_token_fetches_and_caches():
    _token_cache.clear()
    installation_id = 99

    with (
        patch("app.github.app_auth._load_private_key", return_value=TEST_PRIVATE_KEY),
        patch("app.github.app_auth.settings", _mock_settings()),
        patch("app.github.app_auth._get_installation_id", new=AsyncMock(return_value=installation_id)),
        patch("app.github.app_auth._fetch_installation_token", new=AsyncMock(
            return_value=MagicMock(token="ghs_testtoken", expires_at=time.time() + 3600)
        )) as mock_fetch,
    ):
        token = await get_installation_token("org/repo")

    assert token == "ghs_testtoken"
    mock_fetch.assert_called_once()


@pytest.mark.asyncio
async def test_get_installation_token_uses_cache_on_second_call():
    _token_cache.clear()
    installation_id = 42

    with (
        patch("app.github.app_auth._load_private_key", return_value=TEST_PRIVATE_KEY),
        patch("app.github.app_auth.settings", _mock_settings()),
        patch("app.github.app_auth._get_installation_id", new=AsyncMock(return_value=installation_id)),
        patch("app.github.app_auth._fetch_installation_token", new=AsyncMock(
            return_value=MagicMock(token="ghs_cached", expires_at=time.time() + 3600)
        )) as mock_fetch,
    ):
        await get_installation_token("org/repo")
        token = await get_installation_token("org/repo")

    assert token == "ghs_cached"
    assert mock_fetch.call_count == 1  # second call must use cache


@pytest.mark.asyncio
async def test_get_installation_token_refreshes_near_expiry():
    _token_cache.clear()
    installation_id = 77

    with (
        patch("app.github.app_auth._load_private_key", return_value=TEST_PRIVATE_KEY),
        patch("app.github.app_auth.settings", _mock_settings()),
        patch("app.github.app_auth._get_installation_id", new=AsyncMock(return_value=installation_id)),
        patch("app.github.app_auth._fetch_installation_token", new=AsyncMock(
            return_value=MagicMock(token="ghs_refreshed", expires_at=time.time() + 3600)
        )) as mock_fetch,
    ):
        # Seed cache with a nearly-expired token (2 min left < 5 min buffer)
        from app.github.app_auth import _CachedToken
        _token_cache[installation_id] = _CachedToken(
            token="ghs_old", expires_at=time.time() + 120
        )
        token = await get_installation_token("org/repo")

    assert token == "ghs_refreshed"
    mock_fetch.assert_called_once()
