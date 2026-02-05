"""Inference request/response schemas."""

from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    rank: int
    tree_id: int
    tree_key: str
    score: float
    cosine_similarity: float
    distance_m: float
    gt_lat: float
    gt_lon: float


class IdentifyResponse(BaseModel):
    success: bool
    prediction: CandidateResult | None = None
    candidates: list[CandidateResult] = []
    candidates_found: int = 0
    search_radius_m: float
    photo_lat: float
    photo_lon: float
    model_used: str
    method: str = "prototype_matching"


class VerifyRequest(BaseModel):
    tree_id: int
    lat: float
    lon: float


class VerifyResponse(BaseModel):
    success: bool
    tree_id: int
    tree_key: str
    similarity: float
    distance_m: float


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str | None = None
    models_loaded: list[str] = []
    trees_count: int = 0
    prototypes_count: int = 0


class ModelInfoResponse(BaseModel):
    id: int
    name: str
    backbone_name: str
    embedding_dim: int
    input_size: int
    is_active: bool
    is_ensemble: bool
    ensemble_members: list[str] | None = None
