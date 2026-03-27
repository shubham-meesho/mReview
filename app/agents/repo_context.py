import json
import logging

from app.agents.base import BaseAgent
from app.agents.guidelines import CONFIDENCE_THRESHOLD, _build_diff_content, _parse_comments
from app.models.review import Comment, RepoContext, ReviewContext

logger = logging.getLogger(__name__)

# Priority order for including context when total budget is tight
_SECTION_PRIORITY = [
    ("incidents",        "PAST PRODUCTION INCIDENTS (.mreview/incidents.md)"),
    ("anti_patterns",    "TEAM ANTI-PATTERNS (.mreview/anti-patterns.md)"),
    ("review_learnings", "RECURRING REVIEW FLAGS (.mreview/review-learnings.md)"),
    ("architecture",     "ARCHITECTURE & CRITICAL PATHS (.mreview/architecture.md)"),
]

MAX_TOTAL_CONTEXT_CHARS = 12_000

_SYSTEM_PROMPT = """\
You are a context-aware code reviewer with access to this team's documented history for this \
specific repository. Your job is NOT to repeat general code quality issues — other reviewers \
handle language rules and security. Your exclusive focus:

1. Does this code introduce a pattern similar to one that caused a past production incident?
2. Does this code repeat an anti-pattern the team has explicitly documented?
3. Does this code violate an architectural constraint or touch a critical path in a risky way?
4. Does this match a recurring pattern the team always flags in reviews?

Only comment if you find a direct, specific match to the context below. Do NOT invent issues \
that aren't grounded in the provided documents. If nothing matches, return [].

{context_sections}

---

Language: {language}

Each line in the diff is prefixed with its new-file line number: `L0010:` for context/added \
lines, `L----:` for deleted lines. Use exactly that number as the `line` field.

For each match found, output a JSON object:
{{
  "file": "<relative file path>",
  "line": <line number from L{{number}}: prefix, integer>,
  "severity": "<BLOCKER|WARNING|SUGGESTION|NITPICK>",
  "category": "repo-context",
  "message": "<one sentence — what the issue is and which document flagged it>",
  "rationale": "<cite the specific section/incident from the context document that applies>",
  "confidence": <float 0.0-1.0>
}}

Return ONLY a valid JSON array. No prose, no markdown, no code fences.
Only flag issues with confidence >= {threshold}.
"""


def _build_context_sections(repo_context: RepoContext) -> str:
    sections = []
    total_chars = 0

    for field_name, title in _SECTION_PRIORITY:
        content = getattr(repo_context, field_name)
        if not content:
            continue
        available = MAX_TOTAL_CONTEXT_CHARS - total_chars
        if available <= 0:
            break
        if len(content) > available:
            content = content[:available].rsplit("\n", 1)[0] + "\n[truncated]"
        sections.append(f"--- {title} ---\n{content}")
        total_chars += len(content)

    return "\n\n".join(sections) if sections else ""


class RepoContextAgent(BaseAgent):
    """
    Reviews a PR diff against repo-specific context: past incidents, anti-patterns,
    architecture constraints, and recurring review learnings stored in .mreview/.
    No-ops silently when .mreview/ files are absent.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        from app.config import settings
        self._model = model
        self._backend = settings.review_backend

    async def review(self, context: ReviewContext) -> list[Comment]:
        if not context.diff_files:
            return []

        if context.repo_context is None or context.repo_context.is_empty():
            logger.info(
                "PR #%d — no .mreview/ context for %s, RepoContextAgent skipped",
                context.pr_number, context.repo_full_name,
            )
            return []

        context_sections = _build_context_sections(context.repo_context)
        system = _SYSTEM_PROMPT.format(
            context_sections=context_sections,
            language=context.primary_language,
            threshold=CONFIDENCE_THRESHOLD,
        )

        diff_content, total_lines = _build_diff_content(context.diff_files)
        user_message = (
            f"PR #{context.pr_number}: {context.pr_title}\n\n"
            f"{diff_content}"
        )

        logger.info(
            "PR #%d — RepoContextAgent: %d diff lines, context from %s",
            context.pr_number, total_lines, context.repo_context.present_files(),
        )

        raw = await self._call_llm(system, user_message)

        try:
            comments = _parse_comments(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("RepoContextAgent JSON parse failed (%s) — returning no comments", e)
            comments = []

        logger.info(
            "PR #%d — RepoContextAgent: %d context-aware comments",
            context.pr_number, len(comments),
        )
        return comments

    async def _call_llm(self, system: str, user_message: str) -> str:
        if self._backend == "cli":
            return await self._call_via_cli(system, user_message)
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
