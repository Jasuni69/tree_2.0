"""Migrate Excel + model checkpoints → PostgreSQL.

Steps:
  1. Read training_data_cleaned.xlsx → populate trees + photos
  2. Insert model_configs for each model in model_config.yaml
  3. Load model(s), compute prototypes, store in prototypes table
  4. Print verification stats

Usage:
  python scripts/migrate_data.py \
    --data E:\tree_id_2.0\data \
    --images E:\tree_id_2.0\images \
    --model-config E:\tree_id_2.0\model_checkpoints\model_config.yaml \
    --db postgresql+asyncpg://treeid:treeid@localhost:5433/treeid
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from api.db.database import Base
from api.db.models import ModelConfig, Photo, Prototype, Tree
from api.services.model_service import ModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def migrate(
    data_dir: str,
    image_base: str,
    model_config_path: str,
    database_url: str,
    n_prototypes: int = 3,
    outlier_threshold: float = 0.5,
    batch_size: int = 32,
):
    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Connect
    engine = create_async_engine(database_url, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

    # ─── Step 1: Import trees + photos ─────────────────────────────
    df = pd.read_excel(data_dir / "training_data_cleaned.xlsx")
    logger.info("Loaded %d photos, %d unique trees", len(df), df["key"].nunique())

    async with session_factory() as session:
        # Trees
        tree_map = {}  # tree_key → Tree.id
        for key, group in df.groupby("key"):
            row = group.iloc[0]
            address = key.rsplit("|", 1)[0] if "|" in key else key
            tree_num = int(key.rsplit("|", 1)[1]) if "|" in key else 0

            tree = Tree(
                tree_key=key,
                address=address,
                tree_number=tree_num,
                gt_lat=float(row["gt_lat"]),
                gt_lon=float(row["gt_lon"]),
            )
            session.add(tree)
            await session.flush()
            tree_map[key] = tree.id

        logger.info("Inserted %d trees", len(tree_map))

        # Photos - store relative paths so they work in Docker too
        photo_count = 0
        for _, row in df.iterrows():
            photo = Photo(
                tree_id=tree_map[row["key"]],
                file_path=row["image_path"],
                photo_lat=float(row["photo_lat"]) if pd.notna(row.get("photo_lat")) else None,
                photo_lon=float(row["photo_lon"]) if pd.notna(row.get("photo_lon")) else None,
                is_training=True,
            )
            session.add(photo)
            photo_count += 1

        await session.commit()
        logger.info("Inserted %d photos", photo_count)

    # ─── Step 2: Insert model configs ──────────────────────────────
    with open(model_config_path) as f:
        model_yaml = yaml.safe_load(f)

    active_models = model_yaml.get("active_models", [])
    active_mode = model_yaml.get("active_mode", "single")

    async with session_factory() as session:
        model_config_map = {}  # name → ModelConfig.id
        for name, cfg in model_yaml.get("models", {}).items():
            mc = ModelConfig(
                name=name,
                checkpoint_path=cfg["checkpoint_path"],
                backbone_name=cfg.get("backbone_name", "convnext_base"),
                embedding_dim=cfg.get("embedding_dim", 1024),
                input_size=cfg.get("input_size", 224),
                is_active=(name in active_models),
                is_ensemble=(active_mode == "ensemble" and name in active_models),
            )
            session.add(mc)
            await session.flush()
            model_config_map[name] = mc.id

        await session.commit()
        logger.info("Inserted %d model configs", len(model_config_map))

    # ─── Step 3: Compute prototypes ────────────────────────────────
    logger.info("Loading model(s) for prototype computation...")
    model_svc = ModelService()
    model_svc.load_from_yaml(model_config_path)

    if not model_svc.is_loaded():
        logger.warning("No models loaded - skipping prototype computation")
        logger.info("Copy checkpoint files to the paths in model_config.yaml and re-run")
        await engine.dispose()
        return

    from PIL import Image
    from torchvision import transforms

    for model_name in model_svc.active_models:
        mc_id = model_config_map.get(model_name)
        if mc_id is None:
            continue

        logger.info("Computing prototypes with model '%s'...", model_name)
        transform = model_svc.transforms[model_name]

        async with session_factory() as session:
            # Get all trees with photos
            result = await session.execute(
                select(Tree.id, Tree.tree_key).order_by(Tree.id)
            )
            all_trees = result.all()

            proto_total = 0
            for tree_id, tree_key in all_trees:
                # Get training photos
                result = await session.execute(
                    select(Photo.file_path).where(
                        Photo.tree_id == tree_id, Photo.is_training.is_(True)
                    )
                )
                photo_paths = [r[0] for r in result.all()]

                if not photo_paths:
                    continue

                # Load images (paths are relative, resolve against image_base)
                images = []
                for p in photo_paths:
                    try:
                        img = Image.open(image_base / p).convert("RGB")
                        images.append(img)
                    except Exception:
                        pass

                if not images:
                    continue

                # Extract embeddings
                embeddings = model_svc.extract_batch_embeddings(
                    images, model_name=model_name, batch_size=batch_size
                )

                # Outlier filter
                mean_emb = embeddings.mean(axis=0)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
                sims = embeddings @ mean_emb
                keep = sims >= outlier_threshold
                if keep.sum() == 0:
                    keep[sims.argmax()] = True
                filtered = embeddings[keep]

                # K-means
                n = min(n_prototypes, len(filtered))
                if n > 1 and len(filtered) >= 3:
                    km = KMeans(n_clusters=n, random_state=42, n_init=10)
                    km.fit(filtered)
                    protos = [c / (np.linalg.norm(c) + 1e-8) for c in km.cluster_centers_]
                else:
                    m = filtered.mean(axis=0)
                    protos = [m / (np.linalg.norm(m) + 1e-8)]

                # Insert
                for i, vec in enumerate(protos):
                    proto = Prototype(
                        tree_id=tree_id,
                        model_config_id=mc_id,
                        embedding=vec.tolist(),
                        prototype_index=i,
                    )
                    session.add(proto)
                proto_total += len(protos)

            await session.commit()
            logger.info("Model '%s': inserted %d prototypes", model_name, proto_total)

    # ─── Step 4: Verify ────────────────────────────────────────────
    async with session_factory() as session:
        tree_count = (await session.execute(
            select(text("count(*)")).select_from(Tree.__table__)
        )).scalar()
        photo_count = (await session.execute(
            select(text("count(*)")).select_from(Photo.__table__)
        )).scalar()
        proto_count = (await session.execute(
            select(text("count(*)")).select_from(Prototype.__table__)
        )).scalar()

        logger.info("=== Migration Complete ===")
        logger.info("Trees:      %d", tree_count)
        logger.info("Photos:     %d", photo_count)
        logger.info("Prototypes: %d", proto_count)

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Migrate data to PostgreSQL")
    parser.add_argument("--data", default=r"E:\tree_id_2.0\data")
    parser.add_argument("--images", default=r"E:\tree_id_2.0\images")
    parser.add_argument("--model-config", default=r"E:\tree_id_2.0\model_checkpoints\model_config.yaml")
    parser.add_argument("--db", default="postgresql+asyncpg://treeid:treeid@localhost:5433/treeid")
    parser.add_argument("--prototypes", type=int, default=3)
    parser.add_argument("--outlier-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    asyncio.run(migrate(
        args.data, args.images, args.model_config, args.db,
        args.prototypes, args.outlier_threshold, args.batch_size,
    ))


if __name__ == "__main__":
    main()
