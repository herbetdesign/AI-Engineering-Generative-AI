"""Application settings loaded from environment variables and .env."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings used by the FoodOps AI application."""

    app_name: str = "FoodOps AI RAG"
    app_env: str = "development"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-6-luna"
    vector_store_path: Path = Path("data/chroma")
    knowledge_base_path: Path = Path("data/knowledge_base")
    conversation_db_path: Path = Path("data/conversations.sqlite3")
    upload_max_size_bytes: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
