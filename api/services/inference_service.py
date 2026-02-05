"""Inference pipeline: GPS filtering → prototype matching → scoring.

Pipeline:
  1. Photo + GPS arrives
  2. Extract embedding via ModelService
  3. GPS bounding box pre-filter on trees table
  4. Fetch prototypes for candidate trees from pgvector
  5. Score: max(cosine_sim(query, proto)) per tree
  6. Optional proximity boost: alpha * sim + (1-alpha) * proximity
  7. Return ranked candidates
"""

import logging
import math

import numpy as np
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.db.models import Prototype, Tree
from api.services.model_service import model_service

logger = logging.getLogger(__name__)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def gps_bounding_box(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Approximate bounding box for GPS radius filter."""
    # ~111,320 meters per degree latitude
    lat_delta = radius_m / 111_320
    # longitude shrinks with latitude
    lon_delta = radius_m / (111_320 * math.cos(math.radians(lat)))
    return (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta)


async def find_candidate_trees(
    session: AsyncSession, lat: float, lon: float, radius_m: float
) -> list[dict]:
    """Find trees within radius using bounding box pre-filter + haversine."""
    min_lat, max_lat, min_lon, max_lon = gps_bounding_box(lat, lon, radius_m)

    result = await session.execute(
        select(Tree).where(
            Tree.gt_lat.between(min_lat, max_lat),
            Tree.gt_lon.between(min_lon, max_lon),
        )
    )
    trees = result.scalars().all()

    candidates = []
    for tree in trees:
        dist = haversine(lat, lon, tree.gt_lat, tree.gt_lon)
        if dist <= radius_m:
            candidates.append({
                "tree_id": tree.id,
                "tree_key": tree.tree_key,
                "gt_lat": tree.gt_lat,
                "gt_lon": tree.gt_lon,
                "distance_m": dist,
            })

    candidates.sort(key=lambda x: x["distance_m"])
    return candidates


async def fetch_prototypes_for_trees(
    session: AsyncSession, tree_ids: list[int], model_config_id: int
) -> dict[int, list[np.ndarray]]:
    """Fetch prototype embeddings for given trees."""
    if not tree_ids:
        return {}

    result = await session.execute(
        select(Prototype).where(
            Prototype.tree_id.in_(tree_ids),
            Prototype.model_config_id == model_config_id,
        )
    )
    prototypes = result.scalars().all()

    tree_protos: dict[int, list[np.ndarray]] = {}
    for proto in prototypes:
        if proto.tree_id not in tree_protos:
            tree_protos[proto.tree_id] = []
        tree_protos[proto.tree_id].append(np.array(proto.embedding, dtype=np.float32))

    return tree_protos


async def identify_tree(
    session: AsyncSession,
    image: Image.Image,
    lat: float,
    lon: float,
    model_config_id: int,
    radius_m: float | None = None,
    top_k: int | None = None,
    alpha: float | None = None,
) -> dict:
    """Full inference pipeline.

    Returns dict with prediction, candidates, scores.
    """
    radius_m = radius_m or settings.default_gps_radius_m
    top_k = top_k or settings.default_top_k
    alpha = alpha if alpha is not None else settings.default_alpha

    # 1. Extract query embedding
    query = model_service.extract_embedding_auto(image)

    # 2. GPS pre-filter
    candidates = await find_candidate_trees(session, lat, lon, radius_m)
    if not candidates:
        return {
            "success": True,
            "candidates": [],
            "candidates_found": 0,
            "prediction": None,
        }

    # 3. Fetch prototypes
    tree_ids = [c["tree_id"] for c in candidates]
    tree_protos = await fetch_prototypes_for_trees(session, tree_ids, model_config_id)

    # 4. Score each candidate: max cosine similarity to any prototype
    scored = []
    for cand in candidates:
        protos = tree_protos.get(cand["tree_id"], [])
        if not protos:
            continue

        cosine_sim = max(float(np.dot(query, p)) for p in protos)

        # Proximity score: 1 at center, 0 at radius edge
        proximity = 1.0 - (cand["distance_m"] / radius_m) if radius_m > 0 else 1.0

        # Combined score
        score = alpha * cosine_sim + (1 - alpha) * proximity

        scored.append({
            "tree_id": cand["tree_id"],
            "tree_key": cand["tree_key"],
            "gt_lat": cand["gt_lat"],
            "gt_lon": cand["gt_lon"],
            "distance_m": cand["distance_m"],
            "cosine_similarity": cosine_sim,
            "score": score,
        })

    # 5. Rank
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Build response
    ranked = []
    for i, s in enumerate(scored[:top_k]):
        ranked.append({
            "rank": i + 1,
            **s,
        })

    return {
        "success": True,
        "prediction": ranked[0] if ranked else None,
        "candidates": ranked,
        "candidates_found": len(scored),
    }


async def verify_tree(
    session: AsyncSession,
    image: Image.Image,
    tree_id: int,
    model_config_id: int,
) -> dict:
    """Check similarity between photo and specific tree's prototypes."""
    query = model_service.extract_embedding_auto(image)

    tree = await session.get(Tree, tree_id)
    if not tree:
        return {"success": False, "error": "Tree not found"}

    protos = await fetch_prototypes_for_trees(session, [tree_id], model_config_id)
    tree_protos = protos.get(tree_id, [])

    if not tree_protos:
        return {"success": False, "error": "No prototypes for this tree"}

    similarity = max(float(np.dot(query, p)) for p in tree_protos)

    return {
        "success": True,
        "tree_id": tree_id,
        "tree_key": tree.tree_key,
        "similarity": similarity,
    }
