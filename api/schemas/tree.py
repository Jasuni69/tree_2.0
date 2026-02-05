"""Tree CRUD request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class TreeCreate(BaseModel):
    tree_key: str = Field(..., description="Unique key like 'ADDRESS|TREE_NUM'")
    address: str
    tree_number: int
    gt_lat: float
    gt_lon: float
    metadata: dict | None = None


class TreeUpdate(BaseModel):
    address: str | None = None
    tree_number: int | None = None
    gt_lat: float | None = None
    gt_lon: float | None = None
    metadata: dict | None = None


class TreeResponse(BaseModel):
    id: int
    tree_key: str
    address: str
    tree_number: int
    gt_lat: float
    gt_lon: float
    metadata: dict | None
    created_at: datetime
    updated_at: datetime
    photo_count: int = 0
    prototype_count: int = 0

    model_config = {"from_attributes": True}


class TreeListResponse(BaseModel):
    trees: list[TreeResponse]
    total: int
    page: int
    page_size: int
