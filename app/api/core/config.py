from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Intelligent Recruitment Assistant"
    API_V1_PREFIX: str = "/api/v1"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"  # model SBERT nhẹ, phù hợp demo
    DEBUG: bool = True
    RUNTIME_DATA_DIR: str = "Data/Runtime/resumes"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024
    # Normal web extraction is AUTO: try Groq when configured, then fall back
    # to the deterministic offline parser.
    OFFLINE_DEFAULT: bool = False
    groq_api_key: str | None = None

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", "release", "production"}
        return value

    class Config:
        env_file = ".env"

settings = Settings()
