from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

from backend.app.configuration.routers import Routers
from core import settings
from core.database import db_helper

class Server:

    __app: FastAPI

    templates = Jinja2Templates(directory="backend/app/front/templates")
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.security.secret_key)

    def __init__(self, app: FastAPI):

        self.__app = app
        self.__register_routers(app)
        self.__regist_middleware(app)

    @staticmethod
    async def get_db() -> AsyncGenerator[AsyncSession, None]:
        async with db_helper.get_session() as session:
            yield session

    def get_app(self) -> FastAPI:
        return self.__app

    @staticmethod
    def __register_routers(app: FastAPI):

        Routers(Routers._discover_routers()).register(app)

    @staticmethod
    def __regist_middleware(app: FastAPI):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",    # React по умолчанию
                "http://127.0.0.1:5173",    
                "https://agent-trade.ru"
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

