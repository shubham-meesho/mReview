from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class FileDiff:
    filename: str
    status: str  # added, modified, removed, renamed
    additions: int
    deletions: int
    patch: str | None  # raw unified diff hunk; None for binary/large files


@dataclass
class ReviewContext:
    run_id: str
    repo_full_name: str
    pr_number: int
    pr_title: str
    pr_description: str
    primary_language: str
    diff_files: list[FileDiff] = field(default_factory=list)


class Comment(BaseModel):
    file: str
    line: int
    severity: str   # BLOCKER | WARNING | SUGGESTION | NITPICK
    category: str
    message: str
    rationale: str
    confidence: float
