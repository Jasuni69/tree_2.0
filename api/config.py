"""App configuration from environment variables."""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://treeid:treeid@db:5432/treeid"

    # Model config
    model_config_path: str = "/models/model_config.yaml"

    # Photo storage
    photo_storage_path: str = "/photos"
    image_base_path: str = "/images"  # mounted training images for prototype recompute

    # Inference defaults
    default_gps_radius_m: float = 15.0
    default_top_k: int = 5
    default_alpha: float = 0.9  # score = alpha * cosine + (1-alpha) * proximity
    n_prototypes: int = 3
    outlier_threshold: float = 0.5

    # CORS
    cors_origins: list[str] = ["*"]

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "TREEID_", "env_file": ".env"}


settings = Settings()
