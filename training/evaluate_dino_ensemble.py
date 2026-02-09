"""
Evaluate DINOv2 and multi-architecture ensembles with GPS filtering.

Combos tested:
1. DINOv2 only (no GPS)
2. DINOv2 + GPS
3. ConvNeXt + DINOv2 (no GPS)
4. ConvNeXt + DINOv2 + GPS

Run: python training/evaluate_dino_ensemble.py
"""

import torch
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
import math
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model_metric import TreeReIdModel


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


def extract_embeddings(model, dataset, device, batch_size, desc):
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


def build_prototypes(df, emb, valid, train_set, n_prototypes=3, outlier_threshold=0.5):
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
    CONVNEXT_MODEL = r'E:\tree_id_2.0\models\metric_384\best_model.pth'
    DINOV2_MODEL = r'E:\tree_id_2.0\models\metric_dinov2\best_model.pth'
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

    # Train/test split (same as other evals)
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

    # ========== LOAD DINOV2 ==========
    print(f"\nLoading DINOv2-Large...")
    dino_ckpt = torch.load(DINOV2_MODEL, map_location=device)
    dino_config = dino_ckpt.get('config', {})
    dino_input_size = dino_config.get('input_size', 378)

    dinov2 = TreeReIdModel(
        backbone_name=dino_config.get('backbone_name', 'vit_large_patch14_reg4_dinov2'),
        embedding_dim=dino_config.get('embedding_dim', 1024),
        pretrained=False, freeze_stages=0,
        dropout_rate=dino_config.get('dropout_rate', 0.3),
        input_size=dino_input_size,
    )
    dinov2.load_state_dict(dino_ckpt['model_state_dict'], strict=False)
    dinov2 = dinov2.to(device)
    dinov2.eval()
    print(f"DINOv2: input={dino_input_size}, emb={dino_config.get('embedding_dim', 1024)}")
    print(f"DINOv2 best_recall: {dino_ckpt.get('best_recall', 'N/A')}")

    ds_dino = SimpleDataset(image_paths, image_base, input_size=dino_input_size)
    emb_dino, valid_dino = extract_embeddings(dinov2, ds_dino, device, 16, 'DINOv2')

    del dinov2
    torch.cuda.empty_cache()

    # ========== LOAD CONVNEXT ==========
    print(f"\nLoading ConvNeXt-Base (384)...")
    convnext_ckpt = torch.load(CONVNEXT_MODEL, map_location=device)
    convnext_config = convnext_ckpt.get('config', {})
    convnext_input_size = convnext_config.get('input_size', 384)

    convnext = TreeReIdModel(
        backbone_name=convnext_config.get('backbone_name', 'convnext_base'),
        embedding_dim=convnext_config.get('embedding_dim', 1024),
        pretrained=False, freeze_stages=0,
        dropout_rate=convnext_config.get('dropout_rate', 0.3),
    )
    convnext.load_state_dict(convnext_ckpt['model_state_dict'], strict=False)
    convnext = convnext.to(device)
    convnext.eval()
    print(f"ConvNeXt: input={convnext_input_size}, emb={convnext_config.get('embedding_dim', 1024)}")
    print(f"ConvNeXt best_recall: {convnext_ckpt.get('best_recall', 'N/A')}")

    ds_convnext = SimpleDataset(image_paths, image_base, input_size=convnext_input_size)
    emb_convnext, valid_convnext = extract_embeddings(convnext, ds_convnext, device, 32, 'ConvNeXt')

    del convnext
    torch.cuda.empty_cache()

    # ========== COMBINE EMBEDDINGS ==========
    valid = valid_dino & valid_convnext
    print(f"\nValid: {valid.sum()} embeddings")
    print(f"DINOv2 dim: {emb_dino.shape[1]}, ConvNeXt dim: {emb_convnext.shape[1]}")

    # L2-normalize each
    emb_dino_norm = emb_dino / (np.linalg.norm(emb_dino, axis=1, keepdims=True) + 1e-8)
    emb_convnext_norm = emb_convnext / (np.linalg.norm(emb_convnext, axis=1, keepdims=True) + 1e-8)

    # Concat DINOv2 + ConvNeXt
    emb_concat = np.concatenate([emb_dino_norm, emb_convnext_norm], axis=1)
    emb_concat = emb_concat / (np.linalg.norm(emb_concat, axis=1, keepdims=True) + 1e-8)
    print(f"Concatenated dim: {emb_concat.shape[1]}")

    # ========== BUILD PROTOTYPES ==========
    print("\nBuilding prototypes...")
    proto_dino = build_prototypes(df, emb_dino_norm, valid, train_set, N_PROTOTYPES, OUTLIER_THRESHOLD)
    proto_convnext = build_prototypes(df, emb_convnext_norm, valid, train_set, N_PROTOTYPES, OUTLIER_THRESHOLD)
    proto_concat = build_prototypes(df, emb_concat, valid, train_set, N_PROTOTYPES, OUTLIER_THRESHOLD)
    print(f"DINOv2 protos: {len(proto_dino)}, ConvNeXt protos: {len(proto_convnext)}, Concat protos: {len(proto_concat)}")

    # GPS coords
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

    # Address mapping
    address_trees = defaultdict(list)
    for key in proto_concat:
        addr = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[addr].append(key)

    # ========== EVALUATION FUNCTION ==========
    def run_eval(name, query_emb, protos, use_gps):
        results = {'top1': 0, 'top3': 0, 'total': 0}
        trivial = 0

        for idx in test_indices:
            if idx >= len(valid) or not valid[idx]:
                continue
            row = df.iloc[idx]
            true_key = row['key']
            query = query_emb[idx]
            if true_key not in protos:
                continue

            if use_gps:
                # GPS filtering: only consider trees within radius
                plat, plon = row['photo_lat'], row['photo_lon']
                candidates = []
                for k, (tlat, tlon) in tree_coords.items():
                    if k not in protos:
                        continue
                    d = haversine(plat, plon, tlat, tlon)
                    if d <= GPS_RADIUS:
                        candidates.append(k)
            else:
                # Address-based filtering
                addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key
                candidates = [k for k in address_trees.get(addr, []) if k in protos]

            if len(candidates) < 1:
                continue

            if len(candidates) == 1:
                trivial += 1
                if candidates[0] == true_key:
                    results['top1'] += 1
                    results['top3'] += 1
                results['total'] += 1
                continue

            # Score candidates
            sims = []
            for ck in candidates:
                best = max(float(np.dot(query, p)) for p in protos[ck])
                sims.append((ck, best))
            sims.sort(key=lambda x: x[1], reverse=True)

            pred = sims[0][0]
            top3 = [s[0] for s in sims[:3]]

            results['total'] += 1
            if pred == true_key:
                results['top1'] += 1
            if true_key in top3:
                results['top3'] += 1

        return results, trivial

    # ========== RUN ALL CONFIGS ==========
    print(f"\n{'='*70}")
    print(f"Evaluating {len(test_indices)} test photos, GPS radius={GPS_RADIUS}m")
    print(f"{'='*70}")

    configs = [
        ('DINOv2 (no GPS)', emb_dino_norm, proto_dino, False),
        ('DINOv2 + GPS', emb_dino_norm, proto_dino, True),
        ('ConvNeXt (no GPS)', emb_convnext_norm, proto_convnext, False),
        ('ConvNeXt + GPS', emb_convnext_norm, proto_convnext, True),
        ('ConvNeXt + DINOv2 (no GPS)', emb_concat, proto_concat, False),
        ('ConvNeXt + DINOv2 + GPS', emb_concat, proto_concat, True),
    ]

    all_results = []
    for name, emb, protos, use_gps in configs:
        results, trivial = run_eval(name, emb, protos, use_gps)
        if results['total'] > 0:
            top1_acc = results['top1'] / results['total'] * 100
            top3_acc = results['top3'] / results['total'] * 100
        else:
            top1_acc = top3_acc = 0

        all_results.append({
            'name': name,
            'top1': top1_acc,
            'top3': top3_acc,
            'total': results['total'],
            'trivial': trivial
        })

    # Print results table
    print(f"\n{'Model':<30} {'Top-1':>8} {'Top-3':>8} {'N':>6} {'Trivial':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<30} {r['top1']:>7.1f}% {r['top3']:>7.1f}% {r['total']:>6} {r['trivial']:>8}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == '__main__':
    evaluate()
