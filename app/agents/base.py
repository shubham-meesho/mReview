from abc import ABC, abstractmethod

from app.models.review import Comment, ReviewContext


class BaseAgent(ABC):
    @abstractmethod
    async def review(self, context: ReviewContext) -> list[Comment]:
        ...
