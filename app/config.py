from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Conversational Evaluation API"
    API_V1_STR: str = ""
    VERSION: str = "0.1.0"

    EMBEDDER_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    CLASSIFIER_MODEL_DIR: Path = Path("app/models/belief_classifier")
    EVIDENCE_MODEL_DIR: Path = Path("app/models/evidence_extractor")

    NEO4J_URI: str = Field("bolt://neo4j:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field("neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field("test", env="NEO4J_PASSWORD")

    MAX_LATENCY_MS: int = 300

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    """
    Cached singleton to avoid re-parsing env on every request.
    Any part of the codebase can call get_settings().
    """
    return Settings()
