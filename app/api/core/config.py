from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Intelligent Recruitment Assistant"
    API_V1_PREFIX: str = "/api/v1"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"  # model SBERT nhẹ, phù hợp demo
    DEBUG: bool = True

    class Config:
        env_file = ".env"

settings = Settings()