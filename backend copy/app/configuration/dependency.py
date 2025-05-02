from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

from core.database import db_helper

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with db_helper.get_session() as session:
        yield session