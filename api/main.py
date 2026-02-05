"""FastAPI app factory. Startup loads models, shutdown cleans up."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.db.database import close_db, init_db
from api.routers import identify, photos, trees
from api.services.model_service import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()

    logger.info("Loading model(s) from %s", settings.model_config_path)
    config_path = Path(settings.model_config_path)
    if config_path.exists():
        model_service.load_from_yaml(str(config_path))
    else:
        logger.warning("Model config not found at %s - running without models", config_path)

    logger.info("Startup complete")
    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()


app = FastAPI(
    title="Tree Re-ID API",
    description="Tree identification from photos + GPS",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trees.router)
app.include_router(photos.router)
app.include_router(identify.router)


@app.get("/")
async def root():
    return {"service": "tree-reid-api", "version": "2.0.0"}
