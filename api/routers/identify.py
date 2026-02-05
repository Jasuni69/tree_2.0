"""Inference + admin endpoints."""

import io
import logging

import torch
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.db.database import get_session
from api.db.models import ModelConfig, Prototype, Tree
from api.schemas.inference import (
    CandidateResult,
    HealthResponse,
    IdentifyResponse,
    ModelInfoResponse,
    VerifyRequest,
    VerifyResponse,
)
from api.services import inference_service, prototype_service
from api.services.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["inference"])


async def _get_active_model_config(session: AsyncSession) -> ModelConfig:
    result = await session.execute(
        select(ModelConfig).where(ModelConfig.is_active.is_(True)).limit(1)
    )
    mc = result.scalar()
    if not mc:
        raise HTTPException(503, "No active model config in database")
    return mc


@router.post("/identify", response_model=IdentifyResponse)
async def identify_tree(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    radius_m: float = Form(None),
    top_k: int = Form(None),
    alpha: float = Form(None),
    session: AsyncSession = Depends(get_session),
):
    if not model_service.is_loaded():
        raise HTTPException(503, "Model not loaded")

    mc = await _get_active_model_config(session)

    # Read image
    file_bytes = await file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    result = await inference_service.identify_tree(
        session, image, lat, lon, mc.id,
        radius_m=radius_m, top_k=top_k, alpha=alpha,
    )

    # Build response
    candidates = []
    for c in result.get("candidates", []):
        candidates.append(CandidateResult(
            rank=c["rank"],
            tree_id=c["tree_id"],
            tree_key=c["tree_key"],
            score=c["score"],
            cosine_similarity=c["cosine_similarity"],
            distance_m=c["distance_m"],
            gt_lat=c["gt_lat"],
            gt_lon=c["gt_lon"],
        ))

    prediction = candidates[0] if candidates else None

    return IdentifyResponse(
        success=True,
        prediction=prediction,
        candidates=candidates,
        candidates_found=result.get("candidates_found", 0),
        search_radius_m=radius_m or settings.default_gps_radius_m,
        photo_lat=lat,
        photo_lon=lon,
        model_used=model_service.get_active_model_name(),
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify_tree(
    file: UploadFile = File(...),
    tree_id: int = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    session: AsyncSession = Depends(get_session),
):
    if not model_service.is_loaded():
        raise HTTPException(503, "Model not loaded")

    mc = await _get_active_model_config(session)

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    result = await inference_service.verify_tree(session, image, tree_id, mc.id)
    if not result["success"]:
        raise HTTPException(404, result.get("error", "Verification failed"))

    tree = await session.get(Tree, tree_id)
    dist = inference_service.haversine(lat, lon, tree.gt_lat, tree.gt_lon)

    return VerifyResponse(
        success=True,
        tree_id=tree_id,
        tree_key=result["tree_key"],
        similarity=result["similarity"],
        distance_m=dist,
    )


@router.post("/trees/{tree_id}/recompute-prototypes")
async def recompute_tree_prototypes(
    tree_id: int, session: AsyncSession = Depends(get_session)
):
    mc = await _get_active_model_config(session)
    count = await prototype_service.compute_prototypes_for_tree(session, tree_id, mc.id)
    return {"tree_id": tree_id, "prototypes_created": count}


@router.post("/prototypes/recompute-all")
async def recompute_all_prototypes(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    mc = await _get_active_model_config(session)
    # Run in background since this is slow
    background_tasks.add_task(_recompute_all_bg, mc.id)
    return {"status": "started", "model_config_id": mc.id}


async def _recompute_all_bg(model_config_id: int):
    from api.db.database import async_session
    async with async_session() as session:
        result = await prototype_service.recompute_all_prototypes(session, model_config_id)
        logger.info("Recompute all done: %s", result)


@router.get("/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_session)):
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    trees_count = (await session.execute(select(func.count()).select_from(Tree))).scalar() or 0
    proto_count = (await session.execute(select(func.count()).select_from(Prototype))).scalar() or 0

    return HealthResponse(
        status="ok" if model_service.is_loaded() else "no_model",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        models_loaded=model_service.active_models,
        trees_count=trees_count,
        prototypes_count=proto_count,
    )


@router.get("/models", response_model=list[ModelInfoResponse])
async def list_models(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(ModelConfig).order_by(ModelConfig.id))
    configs = result.scalars().all()
    return [
        ModelInfoResponse(
            id=mc.id,
            name=mc.name,
            backbone_name=mc.backbone_name,
            embedding_dim=mc.embedding_dim,
            input_size=mc.input_size,
            is_active=mc.is_active,
            is_ensemble=mc.is_ensemble,
            ensemble_members=mc.ensemble_members,
        )
        for mc in configs
    ]
