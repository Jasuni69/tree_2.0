"""
Enhanced holdout evaluation with TTA and improved prototype building.

Improvements:
1. Test-Time Augmentation: Average embeddings from multiple views
2. Hierarchical prototypes: More prototypes for trees with many photos
3. Weighted prototype scoring: Use top-2 average instead of max

Run: python training/evaluate_enhanced.py --model path/to/best_model.pth
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


class TTADataset(Dataset):
    """Dataset with Test-Time Augmentation support."""
    def __init__(self, image_paths, image_base, base_transform, use_tta=False):
        self.image_paths = image_paths
        self.image_base = Path(image_base)
        self.base_transform = base_transform
        self.use_tta = use_tta

        # TTA transforms
        self.tta_transforms = []
        if use_tta:
            # Center crop
            self.tta_transforms.append(transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            # Horizontal flip
            self.tta_transforms.append(transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            # 5-crop: apply FiveCrop in __getitem__ instead of transform
            # (Lambda not picklable for multiprocessing)
            self._five_crop_resize = transforms.Resize((256, 256))
            self._five_crop = transforms.FiveCrop(224)
            self._five_crop_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_base / self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.use_tta:
                # Apply named transforms (center, flip)
                views = [t(img) for t in self.tta_transforms]
                # Apply 5-crop manually (avoids lambda pickle issue)
                resized = self._five_crop_resize(img)
                crops = self._five_crop(resized)
                for crop in crops:
                    views.append(self._five_crop_post(crop))
                return torch.stack(views), idx, True
            else:
                return self.base_transform(img), idx, True
        except:
            if self.use_tta:
                return torch.zeros(7, 3, 224, 224), idx, False
            else:
                return torch.zeros(3, 224, 224), idx, False


class ImageDataset(Dataset):
    """Simple dataset without TTA."""
    def __init__(self, image_paths, image_base, transform):
        self.image_paths = image_paths
        self.image_base = Path(image_base)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_base / self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, idx, True
        except:
            return torch.zeros(3, 224, 224), idx, False


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def build_hierarchical_prototypes(embeddings, n_photos, outlier_threshold=0.4):
    """
    Build prototypes with hierarchical approach:
    - >15 photos: 5 prototypes
    - >8 photos: 3 prototypes
    - else: 1 prototype (mean)

    Returns list of (prototype, weight) tuples.
    """
    if len(embeddings) == 0:
        return []

    # Filter outliers
    mean = embeddings.mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    sims = embeddings @ mean
    keep = sims >= outlier_threshold
    if keep.sum() == 0:
        keep[sims.argmax()] = True
    filtered = embeddings[keep]

    # Determine number of prototypes
    if n_photos > 15:
        n_protos = min(5, len(filtered))
    elif n_photos > 8:
        n_protos = min(3, len(filtered))
    else:
        n_protos = 1

    if n_protos == 1 or len(filtered) < 3:
        m = filtered.mean(axis=0)
        return [(m / np.linalg.norm(m), 1.0)]

    # KMeans clustering
    km = KMeans(n_clusters=n_protos, random_state=42, n_init=10)
    km.fit(filtered)

    # Weight by cluster size
    labels = km.labels_
    weights = []
    prototypes = []
    for i in range(n_protos):
        cluster_size = (labels == i).sum()
        weights.append(cluster_size)
        proto = km.cluster_centers_[i]
        prototypes.append(proto / np.linalg.norm(proto))

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    return list(zip(prototypes, weights))


def weighted_top2_score(query, prototypes_with_weights):
    """
    Score query against prototypes using weighted average of top-2 similarities.
    """
    if not prototypes_with_weights:
        return 0.0

    sims = []
    for proto, weight in prototypes_with_weights:
        sim = float(np.dot(query, proto))
        sims.append((sim, weight))

    # Sort by similarity descending
    sims.sort(key=lambda x: x[0], reverse=True)

    # Weighted average of top-2
    if len(sims) == 1:
        return sims[0][0] * sims[0][1]
    else:
        return (sims[0][0] * sims[0][1] + sims[1][0] * sims[1][1]) / (sims[0][1] + sims[1][1])


def evaluate(model_path: str, data_dir: str = r'E:\tree_id_2.0\data',
             image_base: str = r'E:\tree_id_2.0\images',
             input_file: str = 'training_data_cleaned.xlsx',
             batch_size: int = 32, val_split: float = 0.15,
             outlier_threshold: float = 0.4,
             radius_m: float = 30, seed: int = 42,
             use_tta: bool = True):

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

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract train embeddings (no TTA for train)
    print(f"\nExtracting train embeddings...")
    train_dataset = ImageDataset(df['image_path'].tolist(), image_base, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                              pin_memory=(device.type == 'cuda'), shuffle=False)

    all_embeddings, valid_mask = [], []
    with torch.no_grad():
        for imgs, idx, valid in tqdm(train_loader, desc=f'Train embeddings ({device})'):
            imgs = imgs.to(device)
            emb = model(imgs)
            all_embeddings.append(emb.cpu())
            valid_mask.extend([v.item() for v in valid])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)
    print(f"Extracted {valid_mask.sum()} valid embeddings (dim={all_embeddings.shape[1]})")

    # Build hierarchical prototypes from train set
    print(f"\nBuilding hierarchical prototypes...")
    tree_prototypes = {}
    tree_photo_counts = df.groupby('key').size().to_dict()

    for key, group in df.groupby('key'):
        indices = [i for i in group.index.tolist()
                   if i in train_set and i < len(valid_mask) and valid_mask[i]]
        if not indices:
            continue

        embs = all_embeddings[indices]
        n_photos = tree_photo_counts.get(key, len(indices))
        protos = build_hierarchical_prototypes(embs, n_photos, outlier_threshold)
        tree_prototypes[key] = protos

    # Print prototype stats
    proto_counts = defaultdict(int)
    for protos in tree_prototypes.values():
        proto_counts[len(protos)] += 1
    print(f"Trees with prototypes: {len(tree_prototypes)}")
    for n, count in sorted(proto_counts.items()):
        print(f"  {n} prototype(s): {count} trees")

    # GPS coords + address mapping
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

    address_trees = defaultdict(list)
    for key in tree_prototypes:
        addr = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[addr].append(key)

    # Extract test embeddings with TTA
    print(f"\nExtracting test embeddings (TTA={use_tta})...")
    test_embeddings = {}

    if use_tta:
        # Use TTA dataset
        test_dataset = TTADataset(df['image_path'].tolist(), image_base, transform, use_tta=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size // 2, num_workers=4,
                                pin_memory=(device.type == 'cuda'), shuffle=False)

        with torch.no_grad():
            for views, indices, valid in tqdm(test_loader, desc=f'Test TTA ({device})'):
                # views: [B, 7, 3, 224, 224]
                B = views.size(0)
                views = views.view(B * 7, 3, 224, 224).to(device)
                embs = model(views)  # [B*7, D]
                embs = embs.view(B, 7, -1)  # [B, 7, D]

                # Average embeddings and re-normalize
                avg_embs = embs.mean(dim=1)  # [B, D]
                avg_embs = F.normalize(avg_embs, p=2, dim=1)

                for i, (idx, v) in enumerate(zip(indices, valid)):
                    if v.item() and idx.item() in test_indices:
                        test_embeddings[idx.item()] = avg_embs[i].cpu().numpy()
    else:
        # Use already extracted embeddings
        for idx in test_indices:
            if idx < len(valid_mask) and valid_mask[idx]:
                test_embeddings[idx] = all_embeddings[idx]

    print(f"Extracted {len(test_embeddings)} test embeddings with TTA")

    # Evaluate with baseline scoring (max)
    print(f"\nEvaluating (baseline scoring)...")
    baseline_results = {k: {'correct': 0, 'total': 0} for k in
                        ['proto_only', 'proto_top3', 'gps_proto', 'gps_proto_top3']}
    trivial_baseline = 0

    for idx in tqdm(test_indices, desc='Baseline eval'):
        if idx not in test_embeddings:
            continue
        row = df.iloc[idx]
        true_key = row['key']
        query = test_embeddings[idx]
        if true_key not in tree_prototypes:
            continue

        addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key
        candidates = address_trees.get(addr, [])

        if len(candidates) < 2:
            trivial_baseline += 1
            for m in baseline_results:
                baseline_results[m]['correct'] += 1
                baseline_results[m]['total'] += 1
            continue

        # Max scoring (baseline)
        sims = []
        for ck in candidates:
            if ck in tree_prototypes:
                best = max(float(np.dot(query, p)) for p, w in tree_prototypes[ck])
            else:
                best = 0.0
            sims.append((ck, best))
        sims.sort(key=lambda x: x[1], reverse=True)

        pred = sims[0][0]
        top3 = [s[0] for s in sims[:3]]
        baseline_results['proto_only']['total'] += 1
        baseline_results['proto_top3']['total'] += 1
        if pred == true_key:
            baseline_results['proto_only']['correct'] += 1
        if true_key in top3:
            baseline_results['proto_top3']['correct'] += 1

        # GPS
        plat, plon = row['photo_lat'], row['photo_lon']
        gps_cands = []
        for k, (tlat, tlon) in tree_coords.items():
            if k not in tree_prototypes:
                continue
            d = haversine(plat, plon, tlat, tlon)
            if d <= radius_m:
                best = max(float(np.dot(query, p)) for p, w in tree_prototypes[k])
                gps_cands.append((k, best))
        if gps_cands:
            gps_cands.sort(key=lambda x: x[1], reverse=True)
            baseline_results['gps_proto']['total'] += 1
            baseline_results['gps_proto_top3']['total'] += 1
            if gps_cands[0][0] == true_key:
                baseline_results['gps_proto']['correct'] += 1
            if true_key in [c[0] for c in gps_cands[:3]]:
                baseline_results['gps_proto_top3']['correct'] += 1

    # Evaluate with enhanced scoring (weighted top-2)
    print(f"\nEvaluating (enhanced scoring)...")
    enhanced_results = {k: {'correct': 0, 'total': 0} for k in
                        ['proto_only', 'proto_top3', 'gps_proto', 'gps_proto_top3']}
    trivial_enhanced = 0
    errors = []

    for idx in tqdm(test_indices, desc='Enhanced eval'):
        if idx not in test_embeddings:
            continue
        row = df.iloc[idx]
        true_key = row['key']
        query = test_embeddings[idx]
        if true_key not in tree_prototypes:
            continue

        addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key
        candidates = address_trees.get(addr, [])

        if len(candidates) < 2:
            trivial_enhanced += 1
            for m in enhanced_results:
                enhanced_results[m]['correct'] += 1
                enhanced_results[m]['total'] += 1
            continue

        # Weighted top-2 scoring
        sims = []
        for ck in candidates:
            if ck in tree_prototypes:
                score = weighted_top2_score(query, tree_prototypes[ck])
            else:
                score = 0.0
            sims.append((ck, score))
        sims.sort(key=lambda x: x[1], reverse=True)

        pred = sims[0][0]
        top3 = [s[0] for s in sims[:3]]
        enhanced_results['proto_only']['total'] += 1
        enhanced_results['proto_top3']['total'] += 1
        if pred == true_key:
            enhanced_results['proto_only']['correct'] += 1
        else:
            true_sim = next(s for k, s in sims if k == true_key)
            errors.append({'true_key': true_key, 'pred_key': pred,
                          'margin': sims[0][1] - true_sim, 'candidates': len(candidates)})
        if true_key in top3:
            enhanced_results['proto_top3']['correct'] += 1

        # GPS
        plat, plon = row['photo_lat'], row['photo_lon']
        gps_cands = []
        for k, (tlat, tlon) in tree_coords.items():
            if k not in tree_prototypes:
                continue
            d = haversine(plat, plon, tlat, tlon)
            if d <= radius_m:
                score = weighted_top2_score(query, tree_prototypes[k])
                gps_cands.append((k, score))
        if gps_cands:
            gps_cands.sort(key=lambda x: x[1], reverse=True)
            enhanced_results['gps_proto']['total'] += 1
            enhanced_results['gps_proto_top3']['total'] += 1
            if gps_cands[0][0] == true_key:
                enhanced_results['gps_proto']['correct'] += 1
            if true_key in [c[0] for c in gps_cands[:3]]:
                enhanced_results['gps_proto_top3']['correct'] += 1

    # Print results
    print("\n" + "=" * 60)
    print("BASELINE RESULTS (max prototype similarity)")
    print("=" * 60)
    print(f"Test photos: {len(test_indices)}, Trivial: {trivial_baseline}\n")
    baseline_accs = {}
    for method, d in baseline_results.items():
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            baseline_accs[method] = acc
            print(f"  {method:20s}: {acc:5.1f}% ({d['correct']}/{d['total']})")

    print("\n" + "=" * 60)
    print("ENHANCED RESULTS (TTA + hierarchical + weighted top-2)")
    print("=" * 60)
    print(f"Test photos: {len(test_indices)}, Trivial: {trivial_enhanced}\n")
    enhanced_accs = {}
    for method, d in enhanced_results.items():
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            enhanced_accs[method] = acc
            delta = acc - baseline_accs.get(method, 0)
            sign = '+' if delta >= 0 else ''
            print(f"  {method:20s}: {acc:5.1f}% ({d['correct']}/{d['total']})  [{sign}{delta:.1f}%]")

    if errors:
        margins = [e['margin'] for e in errors]
        print(f"\nErrors: {len(errors)}, Avg margin: {np.mean(margins):.4f}, Close calls (<0.05): {sum(1 for m in margins if m < 0.05)}")

    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    for method in baseline_accs:
        if method in enhanced_accs:
            delta = enhanced_accs[method] - baseline_accs[method]
            sign = '+' if delta >= 0 else ''
            print(f"  {method:20s}: {sign}{delta:.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\metric\best_model.pth')
    parser.add_argument('--no-tta', action='store_true', help='Disable test-time augmentation')
    args = parser.parse_args()
    evaluate(args.model, use_tta=not args.no_tta)
