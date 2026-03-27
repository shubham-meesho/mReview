import json
import logging

from app.agents.base import BaseAgent
from app.models.review import Comment, ReviewContext
from app.agents.guidelines import (
    CONFIDENCE_THRESHOLD,
    _build_diff_content,
    _parse_comments,
)

logger = logging.getLogger(__name__)

_SECURITY_SYSTEM_PROMPT = """\
You are a security-focused code reviewer. You will be given a pull request diff and must identify \
security vulnerabilities and risks only. Do NOT flag general code quality issues — only flag things \
that could lead to a security incident.

Language: {language}

Security categories to check:
- Injection: SQL injection, command injection, LDAP injection, template injection
- Secrets & credentials: hardcoded API keys, passwords, tokens, private keys in code
- Sensitive data exposure: PII or secrets written to logs, error messages, or responses
- Authentication & authorisation: missing auth checks, privilege escalation, insecure direct object refs
- Deserialization: unsafe deserialization of untrusted input
- SSRF: user-controlled URLs passed to HTTP clients without validation
- Path traversal: user-controlled file paths without sanitisation
- Cryptography: weak algorithms (MD5/SHA1 for integrity), hardcoded IVs or salts
- Dependency confusion / supply chain: suspicious imports or package names
- Race conditions with security impact: TOCTOU on auth or file access

Each line in the diff is prefixed with its new-file line number: `L0010:` for context/added lines, \
`L----:` for deleted lines.
Use exactly the number from the `L{{number}}:` prefix as the `line` field in your output.

For each vulnerability found, output a JSON object with this exact schema:
{{
  "file": "<relative file path>",
  "line": <new-file line number from the L{{number}}: prefix, integer>,
  "severity": "<BLOCKER|WARNING|SUGGESTION|NITPICK>",
  "category": "<naming|error-handling|concurrency|null-safety|resource-leak|logic|security|test-coverage|style|other>",
  "message": "<one sentence, actionable, specific to this code>",
  "rationale": "<why this is a security risk, 1-2 sentences>",
  "confidence": <float 0.0-1.0, how certain you are this is a real vulnerability>
}}

Return ONLY a valid JSON array of such objects. No prose, no markdown, no code fences.
If there are no security issues, return an empty array: []
Only flag issues you are confident about (confidence >= {threshold}).
Focus exclusively on security — ignore style, formatting, and non-security correctness issues.
"""


class SecurityAgent(BaseAgent):
    """
    Reviews a PR diff for security vulnerabilities using Claude.
    Runs in parallel with GuidelinesAgent via the orchestrator.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        from app.config import settings
        self._model = model
        self._backend = settings.review_backend

    async def review(self, context: ReviewContext) -> list[Comment]:
        if not context.diff_files:
            return []

        system = _SECURITY_SYSTEM_PROMPT.format(
            language=context.primary_language,
            threshold=CONFIDENCE_THRESHOLD,
        )

        diff_content, total_lines = _build_diff_content(context.diff_files)
        user_message = (
            f"PR #{context.pr_number}: {context.pr_title}\n\n"
            f"{diff_content}"
        )

        logger.info(
            "PR #%d — SecurityAgent sending %d diff lines (backend=%s)",
            context.pr_number, total_lines, self._backend,
        )

        raw = await self._call_llm(system, user_message)

        try:
            comments = _parse_comments(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("SecurityAgent JSON parse failed (%s) — returning no comments", e)
            comments = []

        logger.info(
            "PR #%d — SecurityAgent: %d security comments",
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

