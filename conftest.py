"""
Set required env vars before any app module is imported.
This runs before all test files, regardless of collection order.
"""
import os

os.environ.setdefault("GITHUB_APP_ID", "123456")
os.environ.setdefault("GITHUB_APP_PRIVATE_KEY_PATH", "./keys/github-app.pem")
os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("TARGET_BRANCH", "develop")
