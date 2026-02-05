"""Model loading, embedding extraction, ensemble support.

Config-driven: reads model_config.yaml, loads model(s) at startup.
Reuses TreeReIdModel from training/model_metric.py.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

# Add training dir so we can import TreeReIdModel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "training"))
from model_metric import TreeReIdModel

logger = logging.getLogger(__name__)


class ModelService:
    """Load models from YAML config, extract embeddings."""

    def __init__(self):
        self.models: dict[str, TreeReIdModel] = {}
        self.transforms: dict[str, transforms.Compose] = {}
        self.configs: dict[str, dict] = {}
        self.active_models: list[str] = []
        self.active_mode: str = "single"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_from_yaml(self, config_path: str):
        """Load model(s) from YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.active_mode = config.get("active_mode", "single")
        active_model_names = config.get("active_models", [])

        for name, model_cfg in config.get("models", {}).items():
            if name not in active_model_names:
                continue
            self._load_model(name, model_cfg)

        self.active_models = [n for n in active_model_names if n in self.models]
        logger.info(
            "Loaded %d model(s): %s (mode=%s)",
            len(self.active_models), self.active_models, self.active_mode,
        )

    def _load_model(self, name: str, cfg: dict):
        """Load single model from checkpoint."""
        checkpoint_path = cfg["checkpoint_path"]
        backbone_name = cfg.get("backbone_name", "convnext_base")
        embedding_dim = cfg.get("embedding_dim", 1024)
        input_size = cfg.get("input_size", 224)

        logger.info("Loading model '%s' from %s", name, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_config = checkpoint.get("config", {})

        model = TreeReIdModel(
            backbone_name=ckpt_config.get("backbone_name", backbone_name),
            embedding_dim=ckpt_config.get("embedding_dim", embedding_dim),
            pretrained=False,
            freeze_stages=0,
            dropout_rate=ckpt_config.get("dropout_rate", 0.3),
            input_size=ckpt_config.get("input_size", input_size),
        )

        state_dict = checkpoint["model_state_dict"]
        filtered = {k: v for k, v in state_dict.items() if "classifier" not in k}
        model.load_state_dict(filtered, strict=False)
        model = model.to(self.device)
        model.eval()

        self.models[name] = model
        self.configs[name] = {
            "backbone_name": model.backbone_name,
            "embedding_dim": model.embedding_dim,
            "input_size": input_size,
            "checkpoint_path": checkpoint_path,
        }

        resize_size = int(input_size * 256 / 224)
        self.transforms[name] = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image, model_name: str | None = None) -> np.ndarray:
        """Extract L2-normalized embedding from a PIL image using one model."""
        if model_name is None:
            model_name = self.active_models[0]

        model = self.models[model_name]
        transform = self.transforms[model_name]

        tensor = transform(image).unsqueeze(0).to(self.device)
        embedding = model(tensor).squeeze(0).cpu().numpy()
        return embedding

    @torch.no_grad()
    def extract_embedding_ensemble(self, image: Image.Image) -> np.ndarray:
        """Extract embedding by averaging across all active models, then re-normalize."""
        embeddings = []
        for name in self.active_models:
            emb = self.extract_embedding(image, model_name=name)
            embeddings.append(emb)

        if len(embeddings) == 1:
            return embeddings[0]

        avg = np.mean(embeddings, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        return avg

    @torch.no_grad()
    def extract_embedding_auto(self, image: Image.Image) -> np.ndarray:
        """Extract embedding using configured mode (single or ensemble)."""
        if self.active_mode == "ensemble" and len(self.active_models) > 1:
            return self.extract_embedding_ensemble(image)
        return self.extract_embedding(image)

    @torch.no_grad()
    def extract_batch_embeddings(
        self, images: list[Image.Image], model_name: str | None = None, batch_size: int = 32
    ) -> np.ndarray:
        """Extract embeddings for a batch of PIL images."""
        if model_name is None:
            model_name = self.active_models[0]

        model = self.models[model_name]
        transform = self.transforms[model_name]

        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(self.device)
            embs = model(tensors).cpu().numpy()
            all_embs.append(embs)

        return np.concatenate(all_embs, axis=0)

    def get_active_model_name(self) -> str:
        if self.active_mode == "ensemble":
            return "ensemble:" + "+".join(self.active_models)
        return self.active_models[0] if self.active_models else "none"

    def is_loaded(self) -> bool:
        return len(self.active_models) > 0


# Singleton
model_service = ModelService()
