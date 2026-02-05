"""Compute and update prototypes per tree.

Reuses logic from inference/compute_embeddings_v2.py:
  1. Gather embeddings for all training photos of a tree
  2. Filter outliers (similarity < threshold to mean)
  3. K-means cluster remaining → N prototypes
  4. Store in prototypes table
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.db.models import ModelConfig, Photo, Prototype, Tree
from api.services.model_service import model_service

_image_base = Path(settings.image_base_path)

logger = logging.getLogger(__name__)


async def compute_prototypes_for_tree(
    session: AsyncSession,
    tree_id: int,
    model_config_id: int,
    n_prototypes: int | None = None,
    outlier_threshold: float | None = None,
):
    """Compute prototypes for one tree using training photos."""
    n_prototypes = n_prototypes or settings.n_prototypes
    outlier_threshold = outlier_threshold or settings.outlier_threshold

    # Get training photos
    result = await session.execute(
        select(Photo).where(Photo.tree_id == tree_id, Photo.is_training.is_(True))
    )
    photos = list(result.scalars().all())

    if not photos:
        logger.warning("Tree %d has no training photos", tree_id)
        return 0

    # Extract embeddings
    images = []
    valid_indices = []
    for i, photo in enumerate(photos):
        try:
            # Resolve relative paths against image_base_path
            fpath = Path(photo.file_path)
            if not fpath.is_absolute():
                fpath = _image_base / fpath
            img = Image.open(fpath).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception:
            logger.warning("Could not open photo %s", photo.file_path)

    if not images:
        return 0

    embeddings = model_service.extract_batch_embeddings(images)

    # Outlier filtering
    mean_emb = embeddings.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    similarities = embeddings @ mean_emb
    keep_mask = similarities >= outlier_threshold

    if keep_mask.sum() == 0:
        keep_mask[similarities.argmax()] = True

    filtered = embeddings[keep_mask]

    # K-means prototypes
    n_actual = min(n_prototypes, len(filtered))
    if n_actual > 1 and len(filtered) >= 3:
        km = KMeans(n_clusters=n_actual, random_state=42, n_init=10)
        km.fit(filtered)
        proto_vectors = [c / (np.linalg.norm(c) + 1e-8) for c in km.cluster_centers_]
    else:
        m = filtered.mean(axis=0)
        proto_vectors = [m / (np.linalg.norm(m) + 1e-8)]

    # Delete old prototypes for this tree+model
    await session.execute(
        delete(Prototype).where(
            Prototype.tree_id == tree_id,
            Prototype.model_config_id == model_config_id,
        )
    )

    # Insert new prototypes
    for i, vec in enumerate(proto_vectors):
        proto = Prototype(
            tree_id=tree_id,
            model_config_id=model_config_id,
            embedding=vec.tolist(),
            prototype_index=i,
        )
        session.add(proto)

    await session.commit()
    return len(proto_vectors)


async def recompute_all_prototypes(session: AsyncSession, model_config_id: int) -> dict:
    """Recompute prototypes for all trees. Returns stats."""
    result = await session.execute(select(Tree.id))
    tree_ids = [row[0] for row in result.all()]

    total = 0
    errors = 0
    for tree_id in tree_ids:
        try:
            count = await compute_prototypes_for_tree(session, tree_id, model_config_id)
            total += count
        except Exception:
            logger.exception("Error computing prototypes for tree %d", tree_id)
            errors += 1

    return {"trees_processed": len(tree_ids), "prototypes_created": total, "errors": errors}
