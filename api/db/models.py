"""ORM models: trees, photos, prototypes, model_configs."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from api.db.database import Base


class Tree(Base):
    __tablename__ = "trees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tree_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    address: Mapped[str] = mapped_column(String(255), nullable=False)
    tree_number: Mapped[int] = mapped_column(Integer, nullable=False)
    gt_lat: Mapped[float] = mapped_column(Float, nullable=False)
    gt_lon: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    photos: Mapped[list["Photo"]] = relationship(back_populates="tree", cascade="all, delete-orphan")
    prototypes: Mapped[list["Prototype"]] = relationship(back_populates="tree", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_trees_address", "address"),
        Index("ix_trees_tree_key", "tree_key"),
        Index("ix_trees_coords", "gt_lat", "gt_lon"),
    )


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tree_id: Mapped[int] = mapped_column(ForeignKey("trees.id", ondelete="CASCADE"), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    photo_lat: Mapped[float | None] = mapped_column(Float)
    photo_lon: Mapped[float | None] = mapped_column(Float)
    is_training: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    tree: Mapped["Tree"] = relationship(back_populates="photos")

    __table_args__ = (
        Index("ix_photos_tree_id", "tree_id"),
    )


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(512), nullable=False)
    backbone_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=1024)
    input_size: Mapped[int] = mapped_column(Integer, nullable=False, default=224)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_ensemble: Mapped[bool] = mapped_column(Boolean, default=False)
    ensemble_members: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    prototypes: Mapped[list["Prototype"]] = relationship(back_populates="model_config")


class Prototype(Base):
    __tablename__ = "prototypes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tree_id: Mapped[int] = mapped_column(ForeignKey("trees.id", ondelete="CASCADE"), nullable=False)
    model_config_id: Mapped[int] = mapped_column(
        ForeignKey("model_configs.id", ondelete="CASCADE"), nullable=False
    )
    embedding = mapped_column(Vector(1024), nullable=False)
    prototype_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    tree: Mapped["Tree"] = relationship(back_populates="prototypes")
    model_config: Mapped["ModelConfig"] = relationship(back_populates="prototypes")

    __table_args__ = (
        UniqueConstraint("tree_id", "model_config_id", "prototype_index", name="uq_tree_model_proto"),
        Index("ix_prototypes_tree_model", "tree_id", "model_config_id"),
    )
