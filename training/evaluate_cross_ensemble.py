"""
Cross-architecture ensemble: EfficientNet-B2 + ConvNeXt-Base (384).
Different architectures see different features — real diversity.

EfficientNet: 1408-dim embeddings (classification backbone)
ConvNeXt: 1024-dim embeddings (metric learning)
Combined: concatenate normalized embeddings → 2432-dim

Run: python training/evaluate_cross_ensemble.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from sklearn.cluster import KMeans
import timm
import math
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model_metric import TreeReIdModel


class TreeClassifier(nn.Module):
    """EfficientNet-based tree classifier (from train.py)."""

    def __init__(self, num_classes, backbone='efficientnet_b0', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        return self.backbone(x)


class SimpleDataset(Dataset):
    def __init__(self, image_paths, image_base, input_size=224):
        self.image_paths = image_paths
        self.image_base = Path(image_base)
        self.input_size = input_size
        resize_size = int(input_size * 256 / 224)

        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_base / self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, idx, True
        except:
            return torch.zeros(3, self.input_size, self.input_size), idx, False


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def extract_embeddings_metric(model, dataset, device, batch_size, desc):
    """Extract embeddings from ConvNeXt metric model."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=(device.type == 'cuda'), shuffle=False)
    all_emb = []
    valid_mask = []
    with torch.no_grad():
        for imgs, idx, valid in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            emb = model(imgs)
            all_emb.append(emb.cpu())
            valid_mask.extend([v.item() for v in valid])
    return torch.cat(all_emb, 0).numpy(), np.array(valid_mask, dtype=bool)


def extract_embeddings_classifier(model, dataset, device, batch_size, desc):
    """Extract embeddings from EfficientNet classifier (feature extractor)."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=(device.type == 'cuda'), shuffle=False)
    all_emb = []
    valid_mask = []
    with torch.no_grad():
        for imgs, idx, valid in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            features = model.extract_features(imgs)
            features = F.normalize(features, p=2, dim=1)
            all_emb.append(features.cpu())
            valid_mask.extend([v.item() for v in valid])
    return torch.cat(all_emb, 0).numpy(), np.array(valid_mask, dtype=bool)


def build_prototypes(df, emb, valid, train_set, n_prototypes=3, outlier_threshold=0.5):
    """Build KMeans prototypes from embeddings."""
    tree_prototypes = {}
    for key, group in df.groupby('key'):
        indices = [i for i in group.index.tolist()
                   if i in train_set and i < len(valid) and valid[i]]
        if not indices:
            continue
        embs = emb[indices]
        mean = embs.mean(axis=0)
        mean = mean / np.linalg.norm(mean)
        sims = embs @ mean
        keep = sims >= outlier_threshold
        if keep.sum() == 0:
            keep[sims.argmax()] = True
        filtered = embs[keep]

        n = min(n_prototypes, len(filtered))
        if n > 1 and len(filtered) >= 3:
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            km.fit(filtered)
            protos = [c / np.linalg.norm(c) for c in km.cluster_centers_]
        else:
            m = filtered.mean(axis=0)
            protos = [m / np.linalg.norm(m)]
        tree_prototypes[key] = protos
    return tree_prototypes


def evaluate():
    # Paths
    EFFNET_MODEL = r'E:\tree_id_2.0\models\model_seed42.pt'
    CONVNEXT_MODEL = r'E:\tree_id_2.0\models\metric_384\best_model.pth'
    data_dir = Path(r'E:\tree_id_2.0\data')
    image_base = Path(r'E:\tree_id_2.0\images')
    input_file = 'training_data_cleaned.xlsx'
    seed = 42
    val_split = 0.15
    GPS_RADIUS = 15.0
    N_PROTOTYPES = 3
    OUTLIER_THRESHOLD = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)

    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees")

    # Same split as other evals
    train_indices, test_indices = [], []
    for key, group in df.groupby('key'):
        indices = group.index.tolist()
        if len(indices) < 2:
            train_indices.extend(indices)
            continue
        np.random.shuffle(indices)
        n_test = max(1, int(len(indices) * val_split))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    train_set = set(train_indices)
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

    image_paths = df['image_path'].tolist()

    # ========== LOAD EFFICIENTNET ==========
    print(f"\nLoading EfficientNet-B2...")
    effnet_ckpt = torch.load(EFFNET_MODEL, map_location=device)
    effnet_config = effnet_ckpt.get('config', {})
    num_classes = effnet_config.get('num_classes', 1365)
    backbone_name = effnet_config.get('backbone', 'efficientnet_b2')
    effnet_input_size = effnet_config.get('img_size', 260)

    effnet = TreeClassifier(num_classes=num_classes, backbone=backbone_name, pretrained=False)
    effnet.load_state_dict(effnet_ckpt['model_state_dict'])
    effnet = effnet.to(device)
    effnet.eval()
    print(f"EfficientNet: {backbone_name}, input={effnet_input_size}, features={effnet.feature_dim}")

    # Extract EfficientNet embeddings
    ds_effnet = SimpleDataset(image_paths, image_base, input_size=effnet_input_size)
    emb_effnet, valid_effnet = extract_embeddings_classifier(effnet, ds_effnet, device, 64, 'EfficientNet')

    del effnet
    torch.cuda.empty_cache()

    # ========== LOAD CONVNEXT ==========
    print(f"\nLoading ConvNeXt-Base (384)...")
    convnext_ckpt = torch.load(CONVNEXT_MODEL, map_location=device)
    convnext_config = convnext_ckpt.get('config', {})
    convnext = TreeReIdModel(
        backbone_name=convnext_config.get('backbone_name', 'convnext_base'),
        embedding_dim=convnext_config.get('embedding_dim', 1024),
        pretrained=False, freeze_stages=0,
        dropout_rate=convnext_config.get('dropout_rate', 0.3),
    )
    convnext.load_state_dict(convnext_ckpt['model_state_dict'], strict=False)
    convnext = convnext.to(device)
    convnext.eval()
    convnext_input_size = convnext_config.get('input_size', 384)
    print(f"ConvNeXt: input={convnext_input_size}, emb={convnext_config.get('embedding_dim', 1024)}")

    ds_convnext = SimpleDataset(image_paths, image_base, input_size=convnext_input_size)
    emb_convnext, valid_convnext = extract_embeddings_metric(convnext, ds_convnext, device, 32, 'ConvNeXt')

    del convnext
    torch.cuda.empty_cache()

    # ========== COMBINE ==========
    valid = valid_effnet & valid_convnext
    print(f"\nValid: {valid.sum()} embeddings")
    print(f"EfficientNet dim: {emb_effnet.shape[1]}, ConvNeXt dim: {emb_convnext.shape[1]}")

    # L2-normalize each
    emb_effnet_norm = emb_effnet / (np.linalg.norm(emb_effnet, axis=1, keepdims=True) + 1e-8)
    emb_convnext_norm = emb_convnext / (np.linalg.norm(emb_convnext, axis=1, keepdims=True) + 1e-8)

    # Concatenate and re-normalize
    emb_concat = np.concatenate([emb_effnet_norm, emb_convnext_norm], axis=1)
    emb_concat = emb_concat / (np.linalg.norm(emb_concat, axis=1, keepdims=True) + 1e-8)
    print(f"Concatenated dim: {emb_concat.shape[1]}")

    # ========== BUILD PROTOTYPES ==========
    print("\nBuilding prototypes...")
    proto_effnet = build_prototypes(df, emb_effnet_norm, valid, train_set, N_PROTOTYPES, OUTLIER_THRESHOLD)
    proto_convnext = build_prototypes(df, emb_convnext_norm, valid, train_set, N_PROTOTYPES, OUTLIER_THRESHOLD)
    proto_concat = build_prototypes(df, emb_concat, valid, train_set, N_PROTOTYPES, OUTLIER_THRESHOLD)
    print(f"EfficientNet protos: {len(proto_effnet)}, ConvNeXt protos: {len(proto_convnext)}, Concat protos: {len(proto_concat)}")

    # GPS coords + address mapping
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

    address_trees = defaultdict(list)
    for key in proto_concat:
        addr = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[addr].append(key)

    # ========== EVALUATE ==========
    configs = [
        ('EfficientNet-B2', emb_effnet_norm, proto_effnet),
        ('ConvNeXt-384', emb_convnext_norm, proto_convnext),
        ('Concat (EffNet+ConvNeXt)', emb_concat, proto_concat),
    ]

    print(f"\nEvaluating {len(test_indices)} test photos, GPS={GPS_RADIUS}m")
    print("=" * 70)

    for name, query_emb, protos in configs:
        results = {k: {'correct': 0, 'total': 0} for k in
                   ['proto_only', 'proto_top3', 'gps_proto', 'gps_proto_top3']}
        trivial = 0

        for idx in test_indices:
            if idx >= len(valid) or not valid[idx]:
                continue
            row = df.iloc[idx]
            true_key = row['key']
            query = query_emb[idx]
            if true_key not in protos:
                continue

            addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key
            candidates = address_trees.get(addr, [])

            if len(candidates) < 2:
                trivial += 1
                for m in results:
                    results[m]['correct'] += 1
                    results[m]['total'] += 1
                continue

            sims = []
            for ck in candidates:
                if ck in protos:
                    best = max(float(np.dot(query, p)) for p in protos[ck])
                else:
                    best = 0.0
                sims.append((ck, best))
            sims.sort(key=lambda x: x[1], reverse=True)

            pred = sims[0][0]
            top3 = [s[0] for s in sims[:3]]
            results['proto_only']['total'] += 1
            results['proto_top3']['total'] += 1
            if pred == true_key:
                results['proto_only']['correct'] += 1
            if true_key in top3:
                results['proto_top3']['correct'] += 1

            # GPS
            plat, plon = row['photo_lat'], row['photo_lon']
            gps_cands = []
            for k, (tlat, tlon) in tree_coords.items():
                if k not in protos:
                    continue
                d = haversine(plat, plon, tlat, tlon)
                if d <= GPS_RADIUS:
                    best = max(float(np.dot(query, p)) for p in protos[k])
                    gps_cands.append((k, best))

            if gps_cands:
                gps_cands.sort(key=lambda x: x[1], reverse=True)
                results['gps_proto']['total'] += 1
                results['gps_proto_top3']['total'] += 1
                if gps_cands[0][0] == true_key:
                    results['gps_proto']['correct'] += 1
                if true_key in [c[0] for c in gps_cands[:3]]:
                    results['gps_proto_top3']['correct'] += 1

        print(f"\n--- {name} ---")
        print(f"Trivial: {trivial}")
        for method, d in results.items():
            if d['total'] > 0:
                acc = d['correct'] / d['total'] * 100
                print(f"  {method:20s}: {acc:5.1f}% ({d['correct']}/{d['total']})")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == '__main__':
    evaluate()
