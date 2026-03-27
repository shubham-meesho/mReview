from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class MethodContext:
    """Full source of a method that contains one or more changed lines."""
    start_line: int   # 1-indexed line in the file where the method begins
    end_line: int     # 1-indexed line in the file where the method ends
    source: str       # complete method text (signature + body)
    truncated: bool = False  # True if the method was capped at MAX_METHOD_LINES


@dataclass
class FileDiff:
    filename: str
    status: str  # added, modified, removed, renamed
    additions: int
    deletions: int
    patch: str | None        # raw unified diff hunk; None for binary/large files
    blob_sha: str | None = None          # SHA of the file blob at PR head commit
    method_contexts: list[MethodContext] = field(default_factory=list)


@dataclass
class RepoContext:
    """
    Repo-specific knowledge fetched from .mreview/ at the root of the repository.
    Each field corresponds to one markdown file; None means the file wasn't found.
    """
    incidents: str | None = None         # .mreview/incidents.md
    review_learnings: str | None = None  # .mreview/review-learnings.md
    architecture: str | None = None      # .mreview/architecture.md
    anti_patterns: str | None = None     # .mreview/anti-patterns.md

    def is_empty(self) -> bool:
        return not any([self.incidents, self.review_learnings,
                        self.architecture, self.anti_patterns])

    def present_files(self) -> list[str]:
        found = []
        if self.incidents:        found.append("incidents.md")
        if self.review_learnings: found.append("review-learnings.md")
        if self.architecture:     found.append("architecture.md")
        if self.anti_patterns:    found.append("anti-patterns.md")
        return found


@dataclass
class ReviewContext:
    run_id: str
    repo_full_name: str
    pr_number: int
    pr_title: str
    pr_description: str
    primary_language: str
    diff_files: list[FileDiff] = field(default_factory=list)
    repo_context: RepoContext | None = None


class Comment(BaseModel):
    file: str
    line: int
    severity: str   # BLOCKER | WARNING | SUGGESTION | NITPICK
    category: str
    message: str
    rationale: str
    confidence: float
