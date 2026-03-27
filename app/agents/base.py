import asyncio
import logging
import os
import subprocess
from abc import ABC, abstractmethod

from app.models.review import Comment, ReviewContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    @abstractmethod
    async def review(self, context: ReviewContext) -> list[Comment]:
        ...

    async def _call_via_cli(self, system: str, user_message: str) -> str:
        """
        Call the `claude` CLI in non-interactive print mode.
        Passes user_message via stdin (not -p flag) to avoid OS arg-size limits
        on large prompts with method context blocks.
        Strips ANTHROPIC_API_KEY from env so the CLI uses stored OAuth credentials.
        """
        def _run() -> str:
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            result = subprocess.run(
                ["/opt/homebrew/bin/claude", "-p", user_message,
                 "--system-prompt", system],
                capture_output=True,
                text=True,
                timeout=300,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"claude CLI exited {result.returncode}: "
                    f"stdout={result.stdout[:200]!r} stderr={result.stderr[:200]!r}"
                )
            output = result.stdout.strip()
            logger.info("CLI raw output (%d chars): %.300s", len(output), output)
            return output

        return await asyncio.to_thread(_run)
