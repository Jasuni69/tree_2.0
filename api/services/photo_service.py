"""Photo storage and EXIF GPS extraction."""

import logging
import shutil
import uuid
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.db.models import Photo, Tree

logger = logging.getLogger(__name__)


def extract_gps_from_exif(image_path: str) -> tuple[float | None, float | None]:
    """Extract GPS lat/lon from image EXIF data."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None, None

        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value

        if not gps_info:
            return None, None

        def to_degrees(value):
            d, m, s = value
            return float(d) + float(m) / 60 + float(s) / 3600

        lat = to_degrees(gps_info["GPSLatitude"])
        if gps_info.get("GPSLatitudeRef", "N") == "S":
            lat = -lat

        lon = to_degrees(gps_info["GPSLongitude"])
        if gps_info.get("GPSLongitudeRef", "E") == "W":
            lon = -lon

        return lat, lon
    except Exception:
        return None, None


async def save_photo(
    session: AsyncSession,
    tree_id: int,
    file_bytes: bytes,
    filename: str,
    photo_lat: float | None = None,
    photo_lon: float | None = None,
    is_training: bool = True,
) -> Photo:
    """Save uploaded photo to disk and create DB record."""
    # Verify tree exists
    tree = await session.get(Tree, tree_id)
    if not tree:
        raise ValueError(f"Tree {tree_id} not found")

    # Save file
    storage = Path(settings.photo_storage_path)
    tree_dir = storage / tree.tree_key.replace("|", "_")
    tree_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(filename).suffix or ".jpg"
    dest_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = tree_dir / dest_name

    dest_path.write_bytes(file_bytes)

    # Try EXIF GPS if not provided
    if photo_lat is None or photo_lon is None:
        exif_lat, exif_lon = extract_gps_from_exif(str(dest_path))
        if photo_lat is None:
            photo_lat = exif_lat
        if photo_lon is None:
            photo_lon = exif_lon

    # Create DB record
    photo = Photo(
        tree_id=tree_id,
        file_path=str(dest_path),
        photo_lat=photo_lat,
        photo_lon=photo_lon,
        is_training=is_training,
    )
    session.add(photo)
    await session.commit()
    await session.refresh(photo)
    return photo


async def delete_photo(session: AsyncSession, photo_id: int) -> bool:
    """Delete photo from disk and DB."""
    photo = await session.get(Photo, photo_id)
    if not photo:
        return False

    # Delete file
    file_path = Path(photo.file_path)
    if file_path.exists():
        file_path.unlink()

    await session.delete(photo)
    await session.commit()
    return True


async def get_photos_for_tree(session: AsyncSession, tree_id: int) -> list[Photo]:
    result = await session.execute(
        select(Photo).where(Photo.tree_id == tree_id).order_by(Photo.created_at)
    )
    return list(result.scalars().all())
