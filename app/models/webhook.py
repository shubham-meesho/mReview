from pydantic import BaseModel


class Repository(BaseModel):
    full_name: str


class PullRequestRef(BaseModel):
    ref: str
    sha: str | None = None


class PullRequest(BaseModel):
    number: int
    title: str
    body: str | None = None
    head: PullRequestRef
    base: PullRequestRef


class WebhookPayload(BaseModel):
    action: str
    pull_request: PullRequest
    repository: Repository
