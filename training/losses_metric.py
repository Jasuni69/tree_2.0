"""
Loss functions for metric learning: SubCenterArcFace + BatchHardTripletLoss.
Adapted from tree_id_new/tree_id/src/losses.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss with online hard mining.
    For each anchor: hardest positive (max dist) and hardest negative (min dist).
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise squared distances
        dot = torch.mm(embeddings, embeddings.t())
        sq_norm = torch.diag(dot)
        distances = sq_norm.unsqueeze(0) - 2.0 * dot + sq_norm.unsqueeze(1)
        distances = F.relu(distances)

        # Masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        not_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)

        # Hardest positive: same label, max distance
        pos_mask = labels_eq & not_self
        masked_pos = distances * pos_mask.float()
        hardest_pos, _ = masked_pos.max(dim=1)
        valid_pos = pos_mask.any(dim=1)

        # Hardest negative: different label, min distance
        neg_mask = ~labels_eq
        max_dist = distances.max().item() + 1.0
        masked_neg = distances + (~neg_mask).float() * max_dist
        hardest_neg, _ = masked_neg.min(dim=1)
        valid_neg = neg_mask.any(dim=1)

        valid = valid_pos & valid_neg
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss[valid].mean()


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace Loss.
    K sub-centers per class to handle intra-class variation (seasons, angles).
    """
    def __init__(self, in_features: int, num_classes: int,
                 num_subcenters: int = 3, margin: float = 0.5, scale: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * num_subcenters, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(torch.pi - margin))
        self.mm = torch.sin(torch.tensor(torch.pi - margin)) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine_all = F.linear(embeddings, weight_norm)

        batch_size = embeddings.size(0)
        cosine_all = cosine_all.view(batch_size, self.num_classes, self.num_subcenters)
        cosine, _ = cosine_all.max(dim=2)

        cos_m = self.cos_m.to(cosine.device, dtype=cosine.dtype)
        sin_m = self.sin_m.to(cosine.device, dtype=cosine.dtype)
        th = self.th.to(cosine.device, dtype=cosine.dtype)
        mm = self.mm.to(cosine.device, dtype=cosine.dtype)

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)
        phi = phi.to(cosine.dtype)

        output = cosine.clone()
        output[range(batch_size), labels] = phi[range(batch_size), labels]
        output *= self.scale

        return F.cross_entropy(output, labels)
