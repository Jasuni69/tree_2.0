"""Photo upload/manage endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import get_session
from api.db.models import Photo
from api.schemas.photo import PhotoListResponse, PhotoResponse
from api.services import photo_service

router = APIRouter(prefix="/api", tags=["photos"])


@router.post("/trees/{tree_id}/photos", response_model=PhotoResponse, status_code=201)
async def upload_photo(
    tree_id: int,
    file: UploadFile = File(...),
    photo_lat: float | None = Form(None),
    photo_lon: float | None = Form(None),
    is_training: bool = Form(True),
    session: AsyncSession = Depends(get_session),
):
    file_bytes = await file.read()
    try:
        photo = await photo_service.save_photo(
            session, tree_id, file_bytes, file.filename or "photo.jpg",
            photo_lat=photo_lat, photo_lon=photo_lon, is_training=is_training,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))

    return PhotoResponse.model_validate(photo)


@router.get("/trees/{tree_id}/photos", response_model=PhotoListResponse)
async def list_photos(tree_id: int, session: AsyncSession = Depends(get_session)):
    photos = await photo_service.get_photos_for_tree(session, tree_id)
    return PhotoListResponse(
        photos=[PhotoResponse.model_validate(p) for p in photos],
        total=len(photos),
    )


@router.get("/photos/{photo_id}/image")
async def serve_photo(photo_id: int, session: AsyncSession = Depends(get_session)):
    photo = await session.get(Photo, photo_id)
    if not photo:
        raise HTTPException(404, "Photo not found")

    file_path = Path(photo.file_path)
    if not file_path.exists():
        raise HTTPException(404, "Photo file not found on disk")

    return FileResponse(str(file_path))


@router.delete("/photos/{photo_id}", status_code=204)
async def delete_photo(photo_id: int, session: AsyncSession = Depends(get_session)):
    deleted = await photo_service.delete_photo(session, photo_id)
    if not deleted:
        raise HTTPException(404, "Photo not found")
