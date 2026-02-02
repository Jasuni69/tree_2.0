"""
Combined best-of-all evaluation: TTA + tight GPS + distance reranking.
Stacks all winning strategies from individual experiments.

Run: python training/evaluate_combined.py
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


class TTADataset(Dataset):
    def __init__(self, image_paths, image_base, use_tta=True, input_size=224):
        self.image_paths = image_paths
        self.image_base = Path(image_base)
        self.use_tta = use_tta
        self.input_size = input_size
        resize_size = int(input_size * 256 / 224)

        self.base_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if use_tta:
            self.flip_transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self._resize = transforms.Resize((resize_size, resize_size))
            self._five_crop = transforms.FiveCrop(input_size)
            self._to_tensor_norm = transforms.Compose([
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
                views = [self.base_transform(img), self.flip_transform(img)]
                resized = self._resize(img)
                crops = self._five_crop(resized)
                for crop in crops:
                    views.append(self._to_tensor_norm(crop))
                return torch.stack(views), idx, True
            else:
                return self.base_transform(img), idx, True
        except:
            if self.use_tta:
                return torch.zeros(7, 3, self.input_size, self.input_size), idx, False
            return torch.zeros(3, self.input_size, self.input_size), idx, False


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def extract_embeddings(model, dataset, device, batch_size, use_tta, desc):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=(device.type == 'cuda'), shuffle=False)
    all_embeddings = []
    valid_mask = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs, idx, valid = batch
            if use_tta:
                # imgs shape: (B, 7, 3, 224, 224)
                B, V, C, H, W = imgs.shape
                imgs_flat = imgs.view(B * V, C, H, W).to(device)
                emb_flat = model(imgs_flat)  # (B*V, dim)
                emb = emb_flat.view(B, V, -1).mean(dim=1)  # average views
                emb = F.normalize(emb, p=2, dim=1)
            else:
                imgs = imgs.to(device)
                emb = model(imgs)
            all_embeddings.append(emb.cpu())
            valid_mask.extend([v.item() for v in valid])

    all_embeddings = torch.cat(all_embeddings, 0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)
    return all_embeddings, valid_mask


def evaluate():
    model_path = r'E:\tree_id_2.0\models\metric_384\best_model.pth'
    data_dir = Path(r'E:\tree_id_2.0\data')
    image_base = Path(r'E:\tree_id_2.0\images')
    input_file = 'training_data_cleaned.xlsx'
    seed = 42
    val_split = 0.15

    # Best params from experiments
    GPS_RADIUS = 15.0
    ALPHA = 0.9  # score = alpha * cosine + (1-alpha) * proximity
    N_PROTOTYPES = 3
    OUTLIER_THRESHOLD = 0.5
    USE_TTA = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)

    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees")

    # Same split
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

    # Load model
    print(f"Loading model from {model_path}")
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
    print(f"Input size: {input_size}x{input_size}")

    # Extract train embeddings (no TTA for prototypes)
    print("\nExtracting train embeddings (no TTA)...")
    train_dataset = TTADataset(df['image_path'].tolist(), image_base, use_tta=False, input_size=input_size)
    train_emb, train_valid = extract_embeddings(model, train_dataset, device, 32, False, 'Train')

    # Extract test embeddings (with TTA)
    print(f"\nExtracting test embeddings (TTA={USE_TTA})...")
    test_dataset = TTADataset(df['image_path'].tolist(), image_base, use_tta=USE_TTA, input_size=input_size)
    test_emb, test_valid = extract_embeddings(model, test_dataset, device,
                                               8 if USE_TTA else 32, USE_TTA, 'Test TTA')

    print(f"Train: {train_valid.sum()} valid, Test: {test_valid.sum()} valid")

    # Build prototypes from train set
    print("\nBuilding prototypes...")
    tree_prototypes = {}
    for key, group in df.groupby('key'):
        indices = [i for i in group.index.tolist()
                   if i in train_set and i < len(train_valid) and train_valid[i]]
        if not indices:
            continue
        embs = train_emb[indices]
        mean = embs.mean(axis=0)
        mean = mean / np.linalg.norm(mean)
        sims = embs @ mean
        keep = sims >= OUTLIER_THRESHOLD
        if keep.sum() == 0:
            keep[sims.argmax()] = True
        filtered = embs[keep]

        n = min(N_PROTOTYPES, len(filtered))
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

    # ========== EVALUATE ALL STRATEGIES ==========
    print(f"\nEvaluating {len(test_indices)} held-out photos...")
    print(f"Config: GPS={GPS_RADIUS}m, alpha={ALPHA}, TTA={USE_TTA}")

    results = {k: {'correct': 0, 'total': 0} for k in [
        'proto_only', 'proto_top3',
        'gps_proto', 'gps_proto_top3',
        'gps_rerank', 'gps_rerank_top3',
    ]}
    trivial = 0
    errors_gps_rerank = []

    # Multi-photo voting
    tree_test_photos = defaultdict(list)

    for idx in tqdm(test_indices, desc='Evaluating'):
        if idx >= len(test_valid) or not test_valid[idx]:
            continue
        row = df.iloc[idx]
        true_key = row['key']
        query = test_emb[idx]
        if true_key not in tree_prototypes:
            continue

        addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key
        candidates = address_trees.get(addr, [])

        # --- Proto only (address-scoped) ---
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
        if true_key in top3:
            results['proto_top3']['correct'] += 1

        # --- GPS proto (tight radius) ---
        plat, plon = row['photo_lat'], row['photo_lon']
        gps_cands = []
        for k, (tlat, tlon) in tree_coords.items():
            if k not in tree_prototypes:
                continue
            d = haversine(plat, plon, tlat, tlon)
            if d <= GPS_RADIUS:
                best = max(float(np.dot(query, p)) for p in tree_prototypes[k])
                gps_cands.append((k, best, d))

        if gps_cands:
            # Pure cosine
            gps_cands_sorted = sorted(gps_cands, key=lambda x: x[1], reverse=True)
            results['gps_proto']['total'] += 1
            results['gps_proto_top3']['total'] += 1
            if gps_cands_sorted[0][0] == true_key:
                results['gps_proto']['correct'] += 1
            if true_key in [c[0] for c in gps_cands_sorted[:3]]:
                results['gps_proto_top3']['correct'] += 1

            # Reranked: alpha * cosine + (1-alpha) * proximity
            reranked = []
            for k, sim, d in gps_cands:
                proximity = 1.0 - (d / GPS_RADIUS)
                score = ALPHA * sim + (1 - ALPHA) * proximity
                reranked.append((k, score))
            reranked.sort(key=lambda x: x[1], reverse=True)

            results['gps_rerank']['total'] += 1
            results['gps_rerank_top3']['total'] += 1
            if reranked[0][0] == true_key:
                results['gps_rerank']['correct'] += 1
            else:
                true_score = next(s for k, s in reranked if k == true_key) if any(k == true_key for k, _ in reranked) else 0
                errors_gps_rerank.append({
                    'true_key': true_key,
                    'pred_key': reranked[0][0],
                    'margin': reranked[0][1] - true_score,
                    'candidates': len(gps_cands),
                })
            if true_key in [c[0] for c in reranked[:3]]:
                results['gps_rerank_top3']['correct'] += 1

            # Track for multi-photo voting
            tree_test_photos[true_key].append(reranked[0][0])

    # Multi-photo voting
    vote_correct = 0
    vote_total = 0
    for true_key, preds in tree_test_photos.items():
        if len(preds) >= 2:
            from collections import Counter
            most_common = Counter(preds).most_common(1)[0][0]
            vote_total += 1
            if most_common == true_key:
                vote_correct += 1

    # ========== RESULTS ==========
    print("\n" + "=" * 70)
    print("COMBINED EVALUATION RESULTS")
    print(f"TTA={USE_TTA}, GPS={GPS_RADIUS}m, alpha={ALPHA}")
    print("=" * 70)
    print(f"Test photos: {len(test_indices)}, Trivial: {trivial}\n")

    for method, d in results.items():
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            print(f"  {method:25s}: {acc:5.1f}% ({d['correct']}/{d['total']})")

    if vote_total > 0:
        print(f"\n  Multi-photo voting:       {vote_correct/vote_total*100:5.1f}% ({vote_correct}/{vote_total}) [trees w/ 2+ photos]")

    if errors_gps_rerank:
        margins = [e['margin'] for e in errors_gps_rerank]
        print(f"\nGPS+rerank errors: {len(errors_gps_rerank)}, "
              f"Avg margin: {np.mean(margins):.4f}, "
              f"Close (<0.01): {sum(1 for m in margins if m < 0.01)}")

    # Comparison to baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO 384 BASELINE (no TTA, 30m GPS, pure cosine)")
    print("=" * 70)
    baseline = {'proto_only': 72.5, 'gps_proto': 78.3, 'gps_proto_top3': 94.4}
    for method, base_acc in baseline.items():
        if method in results and results[method]['total'] > 0:
            new_acc = results[method]['correct'] / results[method]['total'] * 100
            delta = new_acc - base_acc
            print(f"  {method:25s}: {base_acc:.1f}% -> {new_acc:.1f}% ({delta:+.1f}%)")


if __name__ == '__main__':
    evaluate()
