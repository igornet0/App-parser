import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    async_sessionmaker,
    AsyncSession,
)

from core.database.models import Base
from core.settings import settings
from core import data_manager

class Database:

    def __init__(self,
                url: str,
                echo: bool = False,
                echo_pool: bool = False,
                pool_size: int = 5,
                max_overflow: int = 10,
    ) -> None:
        self.engine: AsyncEngine = create_async_engine(
            url=url,
            echo=echo,
            echo_pool=echo_pool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            future=True
        )
        
        self.async_session: AsyncSession = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False
        )

    async def dispose(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.async_session() as session:
            try:
                yield session
            finally:
                await session.close()

    async def create_tables(self):
        from core.database.orm_query import orm_add_coin

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with self.async_session() as session:
            for coin in data_manager.coin_list:
                await orm_add_coin(session, coin)

db_helper = Database(
    url=settings.db.database_url,
    echo=settings.db.echo,
    echo_pool=settings.db.echo_pool,
    pool_size=settings.db.pool_size,
    max_overflow=settings.db.max_overflow
)