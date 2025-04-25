from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging

LOG_DEFAULT_FORMAT = '[%(asctime)s] %(name)-35s:%(lineno)-3d - %(levelname)-7s - %(message)s'

class AppBaseConfig:
    """Базовый класс для конфигурации с общими настройками"""
    case_sensitive = False
    env_file = "./settings/prod.env"
    env_file_encoding = "utf-8"
    env_nested_delimiter="__"
    extra = "ignore"

class LoggingConfig(BaseSettings):
    
    model_config = SettingsConfigDict(**AppBaseConfig.__dict__, 
                                      env_prefix="LOGGING_")
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    format: str = LOG_DEFAULT_FORMAT
    
    access_log: bool = Field(default=True)

    @property
    def log_level(self) -> int:
        return getattr(logging, self.level)

class Config(BaseSettings):

    model_config = SettingsConfigDict(
        **AppBaseConfig.__dict__,
    )

    database_url: str = Field(default=...)

    logging: LoggingConfig = Field(default_factory=LoggingConfig)

settings = Config()