from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.database import db_helper

import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Менеджер жизненного цикла приложения"""
    try:
        logger.info("Initializing database...")
        await db_helper.init_db()
        logger.info("Application startup complete")
        yield
    finally:
        logger.info("Shutting down application...")
        await db_helper.dispose()
        logger.info("Application shutdown complete")