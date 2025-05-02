from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from backend.app.configuration.routers import Routers

class Server:

    __app: FastAPI

    templates = Jinja2Templates(directory="backend/app/front")

    def __init__(self, app: FastAPI):
        self.__app = app
        self.__register_routers(app)
        self.__regist_middleware(app)

    def get_app(self) -> FastAPI:
        return self.__app

    @staticmethod
    def __register_routers(app: FastAPI):

        Routers(Routers._discover_routers()).register(app)

    @staticmethod
    def __regist_middleware(app: FastAPI):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

