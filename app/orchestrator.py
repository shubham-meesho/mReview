import asyncio
import logging

from app.agents.guidelines import GuidelinesAgent
from app.agents.repo_context import RepoContextAgent
from app.agents.security import SecurityAgent
from app.models.review import Comment, ReviewContext

logger = logging.getLogger(__name__)


def _deduplicate(comments: list[Comment]) -> list[Comment]:
    """
    Remove near-duplicate comments: if two comments point at the same
    file + line (within 2 lines) and share the same category, keep only
    the higher-confidence one.
    """
    kept: list[Comment] = []
    for candidate in sorted(comments, key=lambda c: -c.confidence):
        duplicate = any(
            c.file == candidate.file
            and abs(c.line - candidate.line) <= 2
            and c.category == candidate.category
            for c in kept
        )
        if not duplicate:
            kept.append(candidate)

    # Re-sort by file then line for clean output
    kept.sort(key=lambda c: (c.file, c.line))
    return kept


async def run_review(context: ReviewContext) -> list[Comment]:
    """
    Run all agents in parallel and return a merged, deduplicated comment list.
    Agent failures are isolated — one failing agent doesn't abort the review.
    """
    agents = [
        GuidelinesAgent(),
        SecurityAgent(),
        RepoContextAgent(),   # no-ops if no .mreview/ files present in the repo
    ]

    results = await asyncio.gather(
        *[agent.review(context) for agent in agents],
        return_exceptions=True,
    )

    all_comments: list[Comment] = []
    for agent, result in zip(agents, results):
        name = type(agent).__name__
        if isinstance(result, Exception):
            logger.error("%s failed for PR #%d: %s", name, context.pr_number, result)
        else:
            logger.info("%s returned %d comments for PR #%d", name, len(result), context.pr_number)
            all_comments.extend(result)

    deduped = _deduplicate(all_comments)
    logger.info(
        "PR #%d — %d total comments after deduplication (%d before)",
        context.pr_number, len(deduped), len(all_comments),
    )
    return deduped
