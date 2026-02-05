"""Photo upload/manage schemas."""

from datetime import datetime

from pydantic import BaseModel


class PhotoResponse(BaseModel):
    id: int
    tree_id: int
    file_path: str
    photo_lat: float | None
    photo_lon: float | None
    is_training: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class PhotoListResponse(BaseModel):
    photos: list[PhotoResponse]
    total: int
