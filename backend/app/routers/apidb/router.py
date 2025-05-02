from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import db_helper

router = APIRouter(prefix="/api_db", tags=["Api db"])

@router.get("/example")
async def example_endpoint(
    session: AsyncSession = Depends(db_helper.get_session)
):
    # Работа с базой данных
    return {"status": "OK"}