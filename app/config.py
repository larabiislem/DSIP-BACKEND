from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "dsip-backend"
    database_url: str
    model_version: str = "statistical_v1"
    train_days: int = 7
    granularity: str = "H"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
