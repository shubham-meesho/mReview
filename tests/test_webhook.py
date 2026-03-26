import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app  # noqa: E402

client = TestClient(app)

SECRET = "test-secret"

PR_PAYLOAD = {
    "action": "opened",
    "pull_request": {
        "number": 42,
        "title": "Add feature X",
        "body": "Description",
        "head": {"ref": "feature/x"},
        "base": {"ref": "develop"},
    },
    "repository": {"full_name": "org/repo"},
}


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _post(payload: dict, event: str = "pull_request", secret: str = SECRET):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": _sign(body, secret),
            "Content-Type": "application/json",
        },
    )


# --- health ---

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# --- signature verification ---

def test_valid_signature_accepted():
    r = _post(PR_PAYLOAD)
    assert r.status_code == 200


def test_missing_signature_rejected():
    body = json.dumps(PR_PAYLOAD).encode()
    r = client.post(
        "/webhook",
        content=body,
        headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
    )
    assert r.status_code == 403


def test_wrong_signature_rejected():
    r = _post(PR_PAYLOAD, secret="wrong-secret")
    assert r.status_code == 403


# --- event filtering ---

def test_non_pr_event_ignored():
    r = _post(PR_PAYLOAD, event="push")
    assert r.status_code == 200
    assert r.json()["ignored"] is True


def test_action_labeled_ignored():
    payload = {**PR_PAYLOAD, "action": "labeled"}
    r = _post(payload)
    assert r.status_code == 200
    assert r.json()["ignored"] is True


def test_wrong_base_branch_ignored():
    payload = {
        **PR_PAYLOAD,
        "pull_request": {**PR_PAYLOAD["pull_request"], "base": {"ref": "main"}},
    }
    r = _post(payload)
    assert r.status_code == 200
    assert r.json()["ignored"] is True


# --- accepted cases ---

def test_opened_pr_to_target_branch_accepted():
    r = _post(PR_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] is True
    assert data["pr"] == 42
    assert data["repo"] == "org/repo"


def test_synchronize_action_accepted():
    payload = {**PR_PAYLOAD, "action": "synchronize"}
    r = _post(payload)
    assert r.status_code == 200
    assert r.json()["accepted"] is True
