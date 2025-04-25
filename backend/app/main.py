from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core import settings, User
from backend.database import Database

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация БД
    db_instance = Database(settings.database_url)
    await db_instance.create_tables()
    yield
    # Очистка при завершении
    await db_instance.engine.dispose()

app = FastAPI(lifespan=lifespan)
db = Database(settings.database_url)

async def get_db_session():
    async with db.get_session() as session:
        yield session

@app.post("/users/")
async def create_user(
    username: str,
    email: str,
    session: AsyncSession = Depends(get_db_session)
):
    # Проверка уникальности email
    result = await session.execute(select(User).where(User.email == email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    new_user = User(username=username, email=email)
    session.add(new_user)
    await session.commit()
    return new_user

@app.put("/users/{user_id}")
async def update_username(
    user_id: int,
    new_username: str,
    session: AsyncSession = Depends(get_db_session)
):
    result = await session.execute(
        update(User)
        .where(User.id == user_id)
        .values(username=new_username)
        .returning(User)
    )
    updated_user = result.scalars().first()
    
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await session.commit()
    return updated_user

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    session: AsyncSession = Depends(get_db_session)
):
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user