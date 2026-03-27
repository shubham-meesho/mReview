import asyncio
import json
import logging
import pathlib
import subprocess

import anthropic

from app.agents.base import BaseAgent
from app.models.review import Comment, FileDiff, ReviewContext

logger = logging.getLogger(__name__)

# Comments below this threshold are dropped before posting
CONFIDENCE_THRESHOLD = 0.65

# Max diff lines sent to Claude — prevents context overflow on huge PRs
MAX_DIFF_LINES = 6000

# Language-specific rules seeded into the system prompt
_LANG_RULES: dict[str, str] = {
    "Java": """
CORRECTNESS & SAFETY
- Checked exceptions must be caught or declared; never swallow exceptions silently
- Use try-with-resources for anything that implements Closeable/AutoCloseable
- Check all CompletableFuture results; unchecked failures and executor rejections are silent
- String comparison must use .equals(), never ==
- Avoid raw types (List, Map); always parameterise generics
- Resource leaks: close streams, connections, and readers in finally or try-with-resources
- Numeric operations on Long/Integer can be negative; guard percentage checks with Math.abs()
- Throw specific exceptions, not raw RuntimeException — callers need to distinguish failure types

NULL SAFETY
- Prefer Optional<T> over returning null from methods
- Every public method that can return null must document it with @Nullable
- Guard all external inputs and injected config objects with null checks before dereferencing
- Chained calls like a.getB().getC() need null guards at every step

CONCURRENCY
- Avoid Thread.sleep() in application code; use ScheduledExecutorService instead
- Guard shared mutable state with synchronisation; document thread-safety assumptions
- CompletableFuture.runAsync() without a bounded executor uses ForkJoinPool — always pass an executor
- Async tasks that can be rejected by a full executor should be wrapped in try-catch at the call site

DESIGN & MAINTAINABILITY
- Methods with more than one responsibility should be split
- Boolean flag parameters (doX=true) are a code smell; prefer strategy or polymorphism
- Dead code branches (both if/else returning the same value) should be removed
- Magic numbers should be named constants
- Duplicate logic blocks (copy-paste) should be extracted to a shared method
- New public methods on shared services should have Javadoc explaining intent and edge cases

TESTING & OBSERVABILITY
- New branching logic (if/else, feature flags) must have corresponding unit tests
- New classes should have test coverage for both happy path and failure scenarios
- New code paths should emit metrics or structured logs to make them observable in production
- Log levels matter: use log.error only for actionable failures, log.warn for degraded states

SPRING / FRAMEWORK PATTERNS
- Use @Override on every method that overrides or implements an interface
- Prefer constructor injection over field injection for testability
- @Transactional boundaries must be on public methods only
- Executor beans (ExecutorService) should be named and sized explicitly in config

SERIALIZATION & DEPLOYMENT COMPATIBILITY
- Any class that is cached (in-memory, Redis, Memcached) or sent over the wire must have @JsonIgnoreProperties(ignoreUnknown = true); without it, adding a new field will cause deserialization failures on older pods during a rolling deployment
- New fields added to cached or serialized objects must be nullable or have a default value so older serialized payloads (missing the field) can still be deserialized
- Removing or renaming a field from a cached/serialized object is a breaking change; the old name must be kept (or @JsonAlias used) until all pods and caches are drained
- Enum values added to a serialized enum must not break existing persisted values; deserializing an unknown enum value without READ_UNKNOWN_ENUM_VALUES_AS_NULL or a default will throw
""",
    "Kotlin": """
- Avoid !! (non-null assertion); use safe calls (?.) or Elvis operator (?:) instead
- Prefer val over var unless mutation is required
- Use data classes for value objects; override equals/hashCode only when necessary
- Coroutine exceptions must be handled; use CoroutineExceptionHandler or try/catch
- Avoid blocking calls (Thread.sleep, runBlocking) inside coroutine scopes
- Use sealed classes for exhaustive when expressions
- New public functions should have KDoc explaining intent and nullability of parameters
""",
    "Go": """
- Every error return value must be checked; never assign to _
- Avoid panic in library code; return errors instead
- context.Context must be the first argument of functions that do I/O
- Goroutine leaks: ensure every spawned goroutine has a termination path
- Use table-driven tests for functions with multiple input cases
- Mutexes must be value types in structs, never copied after first use
- New exported functions should have godoc comments
""",
    "Python": """
- Bare except clauses must not be used; catch specific exception types
- Mutable default arguments (def f(x=[])) cause subtle bugs; use None sentinel
- Open files must be closed; use context managers (with open(...))
- Type hints should be present on public function signatures
- Avoid broad os.system / subprocess calls without input sanitisation
- New functions with non-obvious behaviour should have docstrings
""",
}

_FALLBACK_RULES = "Apply general software engineering best practices for correctness, safety, and maintainability."

_SYSTEM_PROMPT = """\
You are a thorough, senior code reviewer. Review this pull request diff as carefully as a \
senior engineer would — check every changed file, every new method, every branch.

Language: {language}

Rules to apply:
{rules}

HOW TO REVIEW
- Read the PR description to understand the intent, then check if the implementation matches
- For every new method or class: check correctness, null safety, error handling, and test coverage
- For every new branch (if/else/switch): check all paths are handled, edge cases are covered
- For every new async/concurrent operation: check executor bounds, rejection handling, thread safety
- Flag issues even if you are moderately confident — the confidence score lets us filter later
- Do NOT skip files because they look simple; simple files often hide subtle bugs

Each line in the diff is prefixed with its new-file line number: `L0010:` for context/added \
lines, `L----:` for deleted lines. Use exactly that number as the `line` field.

Some files also include a `[METHOD CONTEXT lines N-M]` block — use it to understand the full \
method signature and logic flow, but always use the L{{number}}: line numbers from the diff \
(not line numbers from the method context block) when setting the `line` field.

For each issue found, output a JSON object:
{{
  "file": "<relative file path>",
  "line": <line number from L{{number}}: prefix, integer>,
  "severity": "<BLOCKER|WARNING|SUGGESTION|NITPICK>",
  "category": "<naming|error-handling|concurrency|null-safety|resource-leak|logic|security|test-coverage|style|other>",
  "message": "<one sentence, actionable, specific to this code>",
  "rationale": "<why this matters, 1-2 sentences>",
  "confidence": <float 0.0-1.0>
}}

Return ONLY a valid JSON array. No prose, no markdown, no code fences.
Include all issues with confidence >= {threshold}. Be thorough — it is better to flag \
something debatable than to silently miss a real bug.
"""


def _annotate_patch(patch: str) -> tuple[str, int]:
    """
    Rewrite a unified diff hunk so every line is prefixed with its
    new-file line number (or '----' for deleted lines).

    Input:
        @@ -10,4 +10,5 @@
         context
        -removed
        +added
         context

    Output:
        @@ -10,4 +10,5 @@
        L0010:  context
        L----: -removed
        L0011: +added
        L0012:  context

    This makes it unambiguous for Claude which line number to use in comments.
    Returns (annotated_patch, lines_processed).
    """
    import re
    output: list[str] = []
    current_line = 0
    lines_processed = 0

    for raw in patch.splitlines():
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if hunk:
            current_line = int(hunk.group(1)) - 1
            output.append(raw)
            continue

        if raw.startswith("-"):
            output.append(f"L----: {raw}")
        else:
            current_line += 1
            output.append(f"L{current_line:04d}: {raw}")
            lines_processed += 1

    return "\n".join(output), lines_processed


def _build_diff_content(files: list[FileDiff]) -> tuple[str, int]:
    """
    Concatenate all file diffs into a single annotated string.
    Each line is prefixed with its new-file line number so Claude
    can produce accurate line references without guessing.
    Returns (content, total_lines). Truncates at MAX_DIFF_LINES.
    """
    lines_used = 0
    parts: list[str] = []
    truncated = False

    for f in files:
        if f.patch is None:
            continue
        header = f"### {f.filename} ({f.status}, +{f.additions}/-{f.deletions})\n"
        annotated, file_lines = _annotate_patch(f.patch)

        if lines_used + file_lines > MAX_DIFF_LINES:
            # Re-annotate with truncation
            remaining = MAX_DIFF_LINES - lines_used
            # Trim to roughly `remaining` non-deleted lines
            trimmed_lines: list[str] = []
            kept = 0
            for line in annotated.splitlines():
                trimmed_lines.append(line)
                if not line.startswith("L----"):
                    kept += 1
                if kept >= remaining:
                    break
            annotated = "\n".join(trimmed_lines)
            truncated = True
            file_lines = kept

        file_block = header + annotated

        # Append enclosing method source blocks so Claude sees the full method
        for mc in f.method_contexts:
            trunc_note = " [truncated]" if mc.truncated else ""
            file_block += (
                f"\n\n[METHOD CONTEXT lines {mc.start_line}-{mc.end_line}{trunc_note}]\n"
                f"{mc.source}\n"
                f"[END METHOD CONTEXT]"
            )

        parts.append(file_block)
        lines_used += file_lines

        if truncated:
            parts.append("\n[diff truncated — exceeded max reviewable lines]")
            break

    return "\n\n".join(parts), lines_used


def _parse_comments(raw: str) -> list[Comment]:
    """
    Parse Claude's response into a list of Comment objects.
    Strips markdown fences defensively before parsing.
    """
    text = raw.strip()
    # Strip markdown code fences if Claude added them despite instructions
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Claude sometimes copies the L0148: prefix format into the line number,
    # producing "line": 0148 which is invalid JSON (leading zeros).  Strip them.
    import re
    text = re.sub(r'"line":\s*0+(\d)', r'"line": \1', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Response may be truncated mid-array — recover all complete objects
        last_close = text.rfind("}")
        if last_close != -1:
            recovered = text[: last_close + 1] + "]"
            # Make sure it starts with '[', accounting for possible leading whitespace
            start = text.find("[")
            if start != -1:
                recovered = text[start : last_close + 1] + "]"
            try:
                data = json.loads(recovered)
                logger.warning("Recovered %d chars of truncated JSON array", len(recovered))
            except json.JSONDecodeError:
                raise  # re-raise original-ish error
        else:
            raise
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")

    comments: list[Comment] = []
    for item in data:
        try:
            c = Comment.model_validate(item)
            if c.confidence >= CONFIDENCE_THRESHOLD:
                comments.append(c)
        except Exception as e:
            logger.warning("Skipping invalid comment object: %s — %s", item, e)

    return comments


class GuidelinesAgent(BaseAgent):
    """
    Reviews a PR diff against language-specific guidelines using Claude.

    Flow:
      1. Build system prompt with language-specific rules
      2. Send diff to Claude (via API or CLI subprocess), request structured JSON output
      3. Parse response; if JSON is invalid, retry once with a correction prompt
      4. Filter out low-confidence comments
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        from app.config import settings
        self._client = anthropic.Anthropic()
        self._model = model
        self._backend = settings.review_backend  # "api" or "cli"

    async def review(self, context: ReviewContext) -> list[Comment]:
        if not context.diff_files:
            logger.info("PR #%d has no reviewable files — skipping", context.pr_number)
            return []

        lang = context.primary_language
        rules = _LANG_RULES.get(lang, _FALLBACK_RULES)
        system = _SYSTEM_PROMPT.format(
            language=lang,
            rules=rules.strip(),
            threshold=CONFIDENCE_THRESHOLD,
        )

        diff_content, total_lines = _build_diff_content(context.diff_files)
        description_block = (
            f"PR Description:\n{context.pr_description.strip()}\n\n"
            if context.pr_description and context.pr_description.strip()
            else ""
        )
        user_message = (
            f"PR #{context.pr_number}: {context.pr_title}\n\n"
            f"{description_block}"
            f"{diff_content}"
        )

        logger.info(
            "PR #%d — sending %d diff lines to %s (backend=%s)",
            context.pr_number, total_lines, self._model, self._backend,
        )

        # Dump prompt to file so it can be manually inspected
        prompt_path = self._dump_prompt(context.pr_number, system, user_message)
        logger.info("PR #%d — prompt written to %s", context.pr_number, prompt_path)

        raw = await self._call_llm(system, user_message)

        # Attempt to parse; retry once if malformed
        try:
            comments = _parse_comments(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("JSON parse failed (%s) — retrying with correction prompt", e)
            comments = await self._retry_parse(system, user_message, raw)

        logger.info(
            "PR #%d — %d comments after confidence filter (threshold=%.2f)",
            context.pr_number, len(comments), CONFIDENCE_THRESHOLD,
        )
        return comments

    async def _call_llm(self, system: str, user_message: str) -> str:
        """Dispatch to API or CLI backend."""
        if self._backend == "cli":
            return await self._call_via_cli(system, user_message)
        return self._call_via_api(system, user_message)

    def _call_via_api(self, system: str, user_message: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


    def build_prompt(self, context: ReviewContext) -> tuple[str, str]:
        """Return (system_prompt, user_message) for this context. Used by /review/inject."""
        lang = context.primary_language
        rules = _LANG_RULES.get(lang, _FALLBACK_RULES)
        system = _SYSTEM_PROMPT.format(
            language=lang,
            rules=rules.strip(),
            threshold=CONFIDENCE_THRESHOLD,
        )
        diff_content, _ = _build_diff_content(context.diff_files)
        user_message = f"PR #{context.pr_number}: {context.pr_title}\n\n{diff_content}"
        return system, user_message

    def _dump_prompt(self, pr_number: int, system: str, user_message: str) -> str:
        """Write the full prompt to prompts/pr_{number}.txt and return the path."""
        prompts_dir = pathlib.Path("prompts")
        prompts_dir.mkdir(exist_ok=True)
        path = prompts_dir / f"pr_{pr_number}.txt"
        path.write_text(
            f"=== SYSTEM PROMPT ===\n{system}\n\n"
            f"=== USER MESSAGE ===\n{user_message}\n",
            encoding="utf-8",
        )
        return str(path)

    async def _retry_parse(
        self,
        system: str,
        original_user_msg: str,
        bad_response: str,
    ) -> list[Comment]:
        """Send a follow-up asking Claude to fix its malformed JSON."""
        correction = (
            f"{original_user_msg}\n\n"
            f"[Previous response was not valid JSON]\n{bad_response}\n\n"
            "Your response was not valid JSON. Return only the JSON array, no prose, no markdown."
        )
        if self._backend == "cli":
            raw = await self._call_via_cli(system, correction)
        else:
            retry_response = self._client.messages.create(
                model=self._model,
                max_tokens=8192,
                system=system,
                messages=[
                    {"role": "user", "content": original_user_msg},
                    {"role": "assistant", "content": bad_response},
                    {"role": "user", "content": "Your response was not valid JSON. Return only the JSON array, no prose, no markdown."},
                ],
            )
            raw = retry_response.content[0].text
        try:
            return _parse_comments(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Retry parse also failed: %s — returning no comments", e)
            return []
