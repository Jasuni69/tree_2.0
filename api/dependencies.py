"""FastAPI Depends() for DB sessions and model service."""

from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import get_session
from api.services.model_service import ModelService, model_service


async def get_db() -> AsyncSession:
    async for session in get_session():
        yield session


def get_model_service() -> ModelService:
    return model_service
