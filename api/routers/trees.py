"""Tree CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import get_session
from api.db.models import Photo, Prototype, Tree
from api.schemas.tree import TreeCreate, TreeListResponse, TreeResponse, TreeUpdate

router = APIRouter(prefix="/api/trees", tags=["trees"])


def _tree_to_response(tree: Tree, photo_count: int = 0, prototype_count: int = 0) -> TreeResponse:
    return TreeResponse(
        id=tree.id,
        tree_key=tree.tree_key,
        address=tree.address,
        tree_number=tree.tree_number,
        gt_lat=tree.gt_lat,
        gt_lon=tree.gt_lon,
        metadata=tree.metadata_,
        created_at=tree.created_at,
        updated_at=tree.updated_at,
        photo_count=photo_count,
        prototype_count=prototype_count,
    )


@router.get("", response_model=TreeListResponse)
async def list_trees(
    address: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_m: float = 100.0,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    query = select(Tree)

    if address:
        query = query.where(Tree.address.ilike(f"%{address}%"))

    if lat is not None and lon is not None:
        # Approximate bounding box filter
        lat_delta = radius_m / 111_320
        import math
        lon_delta = radius_m / (111_320 * math.cos(math.radians(lat)))
        query = query.where(
            Tree.gt_lat.between(lat - lat_delta, lat + lat_delta),
            Tree.gt_lon.between(lon - lon_delta, lon + lon_delta),
        )

    # Count
    count_q = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_q)).scalar() or 0

    # Paginate
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Tree.id)
    result = await session.execute(query)
    trees = result.scalars().all()

    # Get counts
    responses = []
    for tree in trees:
        pc = (await session.execute(
            select(func.count()).where(Photo.tree_id == tree.id)
        )).scalar() or 0
        prc = (await session.execute(
            select(func.count()).where(Prototype.tree_id == tree.id)
        )).scalar() or 0
        responses.append(_tree_to_response(tree, pc, prc))

    return TreeListResponse(trees=responses, total=total, page=page, page_size=page_size)


@router.get("/{tree_id}", response_model=TreeResponse)
async def get_tree(tree_id: int, session: AsyncSession = Depends(get_session)):
    tree = await session.get(Tree, tree_id)
    if not tree:
        raise HTTPException(404, "Tree not found")

    pc = (await session.execute(
        select(func.count()).where(Photo.tree_id == tree.id)
    )).scalar() or 0
    prc = (await session.execute(
        select(func.count()).where(Prototype.tree_id == tree.id)
    )).scalar() or 0

    return _tree_to_response(tree, pc, prc)


@router.post("", response_model=TreeResponse, status_code=201)
async def create_tree(body: TreeCreate, session: AsyncSession = Depends(get_session)):
    # Check uniqueness
    existing = await session.execute(
        select(Tree).where(Tree.tree_key == body.tree_key)
    )
    if existing.scalar():
        raise HTTPException(409, f"Tree with key '{body.tree_key}' already exists")

    tree = Tree(
        tree_key=body.tree_key,
        address=body.address,
        tree_number=body.tree_number,
        gt_lat=body.gt_lat,
        gt_lon=body.gt_lon,
        metadata_=body.metadata,
    )
    session.add(tree)
    await session.commit()
    await session.refresh(tree)
    return _tree_to_response(tree)


@router.put("/{tree_id}", response_model=TreeResponse)
async def update_tree(
    tree_id: int, body: TreeUpdate, session: AsyncSession = Depends(get_session)
):
    tree = await session.get(Tree, tree_id)
    if not tree:
        raise HTTPException(404, "Tree not found")

    for field, value in body.model_dump(exclude_unset=True).items():
        if field == "metadata":
            setattr(tree, "metadata_", value)
        else:
            setattr(tree, field, value)

    await session.commit()
    await session.refresh(tree)
    return _tree_to_response(tree)


@router.delete("/{tree_id}", status_code=204)
async def delete_tree(tree_id: int, session: AsyncSession = Depends(get_session)):
    tree = await session.get(Tree, tree_id)
    if not tree:
        raise HTTPException(404, "Tree not found")

    await session.delete(tree)
    await session.commit()
