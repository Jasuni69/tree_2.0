"""
Tree re-identification model with metric learning.
Supports ConvNeXt (CNN) and DINOv2 (ViT) backbones.
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


# Backbone feature dimensions
BACKBONE_DIMS = {
    'convnext_tiny': 768,
    'convnext_small': 768,
    'convnext_base': 1024,
    'convnext_large': 1536,
    'vit_base_patch14_reg4_dinov2': 768,
    'vit_large_patch14_reg4_dinov2': 1024,
    'vit_giant_patch14_reg4_dinov2': 1536,
    'vit_base_patch14_dinov2': 768,
    'vit_large_patch14_dinov2': 1024,
    'vit_giant_patch14_dinov2': 1536,
    'vit_small_patch14_reg4_dinov2': 384,
    'vit_small_patch14_dinov2': 384,
}


def _is_vit(backbone_name: str) -> bool:
    return backbone_name.startswith('vit_')


class TreeReIdModel(nn.Module):
    """
    Tree Re-ID model: backbone + pooling + embedding head.
    ConvNeXt uses GeM pooling (4D features).
    DINOv2 ViT outputs 2D features directly (CLS token).
    Returns L2-normalized embeddings.
    """

    BACKBONE_DIMS = BACKBONE_DIMS

    def __init__(self, backbone_name: str = 'convnext_base',
                 embedding_dim: int = 1024, pretrained: bool = True,
                 freeze_stages: int = 2, dropout_rate: float = 0.3,
                 input_size: int = 224):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.is_vit = _is_vit(backbone_name)
        self.backbone_dim = BACKBONE_DIMS[backbone_name]

        if self.is_vit:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained,
                num_classes=0, img_size=input_size
            )
            self.pool = nn.Identity()
        else:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained,
                num_classes=0, global_pool=''
            )
            self.pool = GeM()

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

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
        if self.is_vit:
            # Freeze patch embed + first N transformer blocks
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, 'blocks'):
                n_blocks = len(self.backbone.blocks)
                freeze_n = min(num_stages * (n_blocks // 4), n_blocks)
                for i in range(freeze_n):
                    for param in self.backbone.blocks[i].parameters():
                        param.requires_grad = False
        else:
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
        if not self.is_vit:
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
