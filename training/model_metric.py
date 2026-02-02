"""
ConvNeXt-Base model for tree re-identification with metric learning.
Adapted from tree_id_new/tree_id/src/tree_reid_model_convnext.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p)


class TreeReIdModel(nn.Module):
    """
    Tree Re-ID model: ConvNeXt backbone + GeM pooling + embedding head.
    Returns L2-normalized embeddings.
    """

    BACKBONE_DIMS = {
        'convnext_tiny': 768,
        'convnext_small': 768,
        'convnext_base': 1024,
        'convnext_large': 1536,
    }

    def __init__(self, backbone_name: str = 'convnext_base',
                 embedding_dim: int = 1024, pretrained: bool = True,
                 freeze_stages: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.backbone_dim = self.BACKBONE_DIMS[backbone_name]

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            num_classes=0, global_pool=''
        )

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

        self.pool = GeM()

        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, embedding_dim)
        )

        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_stages(self, num_stages: int):
        if num_stages >= 1:
            for param in self.backbone.stem.parameters():
                param.requires_grad = False
        for i in range(min(num_stages - 1, 4)):
            for param in self.backbone.stages[i].parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @classmethod
    def from_pretrained_partial(cls, checkpoint_path: str, device: str = 'cuda'):
        """Load backbone + embedding head from checkpoint. No classifier."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})

        if isinstance(config, dict) and 'model' in config:
            model_cfg = config['model']
        else:
            model_cfg = config

        model = cls(
            backbone_name=model_cfg.get('backbone_name', 'convnext_base'),
            embedding_dim=model_cfg.get('embedding_dim', 1024),
            pretrained=False,
            freeze_stages=0,
            dropout_rate=model_cfg.get('dropout_rate', 0.3),
        )

        # Load everything except classifier
        filtered = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys: {missing}")

        model = model.to(device)
        return model
