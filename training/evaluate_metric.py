"""
Holdout evaluation for the metric learning ConvNeXt model.
Same logic as holdout_eval.py but loads our trained model.

Run: python training/evaluate_metric.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from sklearn.cluster import KMeans
import math
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model_metric import TreeReIdModel


class ImageDataset(Dataset):
    def __init__(self, image_paths, image_base, transform, input_size=224):
        self.image_paths = image_paths
        self.image_base = Path(image_base)
        self.transform = transform
        self.input_size = input_size

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


def evaluate(model_path: str, data_dir: str = r'E:\tree_id_2.0\data',
             image_base: str = r'E:\tree_id_2.0\images',
             input_file: str = 'training_data_cleaned.xlsx',
             batch_size: int = 64, val_split: float = 0.15,
             n_prototypes: int = 3, outlier_threshold: float = 0.5,
             radius_m: float = 30, seed: int = 42):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    np.random.seed(seed)

    data_dir = Path(data_dir)
    image_base = Path(image_base)

    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees")

    # Stratified split
    train_indices, test_indices = [], []
    trees_skipped = 0
    for key, group in df.groupby('key'):
        indices = group.index.tolist()
        if len(indices) < 2:
            train_indices.extend(indices)
            trees_skipped += 1
            continue
        np.random.shuffle(indices)
        n_test = max(1, int(len(indices) * val_split))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}, Single-photo trees: {trees_skipped}")
    train_set = set(train_indices)

    # Load model
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})

    model = TreeReIdModel(
        backbone_name=config.get('backbone_name', 'convnext_base'),
        embedding_dim=config.get('embedding_dim', 1024),
        pretrained=False, freeze_stages=0,
        dropout_rate=config.get('dropout_rate', 0.3),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    input_size = config.get('input_size', 224)
    resize_size = int(input_size * 256 / 224)
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"Input size: {input_size}x{input_size}")

    # Extract embeddings
    print(f"\nExtracting embeddings...")
    dataset = ImageDataset(df['image_path'].tolist(), image_base, transform, input_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=(device.type == 'cuda'), shuffle=False)

    all_embeddings, valid_mask = [], []
    with torch.no_grad():
        for imgs, idx, valid in tqdm(loader, desc=f'Extracting ({device})'):
            imgs = imgs.to(device)
            emb = model(imgs)
            all_embeddings.append(emb.cpu())
            valid_mask.extend([v.item() for v in valid])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)
    print(f"Extracted {valid_mask.sum()} valid embeddings (dim={all_embeddings.shape[1]})")

    # Build prototypes from train set
    print(f"\nBuilding prototypes...")
    tree_prototypes = {}
    for key, group in df.groupby('key'):
        indices = [i for i in group.index.tolist()
                   if i in train_set and i < len(valid_mask) and valid_mask[i]]
        if not indices:
            continue

        embs = all_embeddings[indices]
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

    print(f"Trees with prototypes: {len(tree_prototypes)}")

    # GPS coords + address mapping
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

    address_trees = defaultdict(list)
    for key in tree_prototypes:
        addr = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[addr].append(key)

    # Evaluate
    print(f"\nEvaluating {len(test_indices)} held-out photos...")
    results = {k: {'correct': 0, 'total': 0} for k in
               ['proto_only', 'proto_top3', 'gps_proto', 'gps_proto_top3']}
    trivial = 0
    errors = []

    for idx in tqdm(test_indices, desc='Evaluating'):
        if idx >= len(valid_mask) or not valid_mask[idx]:
            continue
        row = df.iloc[idx]
        true_key = row['key']
        query = all_embeddings[idx]
        if true_key not in tree_prototypes:
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
            if ck in tree_prototypes:
                best = max(float(np.dot(query, p)) for p in tree_prototypes[ck])
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
        else:
            true_sim = next(s for k, s in sims if k == true_key)
            errors.append({'true_key': true_key, 'pred_key': pred,
                           'margin': sims[0][1] - true_sim, 'candidates': len(candidates)})
        if true_key in top3:
            results['proto_top3']['correct'] += 1

        # GPS
        plat, plon = row['photo_lat'], row['photo_lon']
        gps_cands = []
        for k, (tlat, tlon) in tree_coords.items():
            if k not in tree_prototypes:
                continue
            d = haversine(plat, plon, tlat, tlon)
            if d <= radius_m:
                best = max(float(np.dot(query, p)) for p in tree_prototypes[k])
                gps_cands.append((k, best))
        if gps_cands:
            gps_cands.sort(key=lambda x: x[1], reverse=True)
            results['gps_proto']['total'] += 1
            results['gps_proto_top3']['total'] += 1
            if gps_cands[0][0] == true_key:
                results['gps_proto']['correct'] += 1
            if true_key in [c[0] for c in gps_cands[:3]]:
                results['gps_proto_top3']['correct'] += 1

    # Print results
    print("\n" + "=" * 60)
    print("HOLDOUT RESULTS - Metric Learning ConvNeXt")
    print("=" * 60)
    print(f"Test photos: {len(test_indices)}, Trivial: {trivial}\n")
    for method, d in results.items():
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            print(f"  {method:20s}: {acc:5.1f}% ({d['correct']}/{d['total']})")

    if errors:
        margins = [e['margin'] for e in errors]
        print(f"\nErrors: {len(errors)}, Avg margin: {np.mean(margins):.4f}, Close calls (<0.05): {sum(1 for m in margins if m < 0.05)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\metric\best_model.pth')
    args = parser.parse_args()
    evaluate(args.model)
