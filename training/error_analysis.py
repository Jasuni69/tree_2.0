"""
Deep error analysis on the 617 failures.
Finds patterns: which addresses fail most, margin distributions,
candidate counts, per-tree error rates, photo counts vs accuracy.

Run: python training/error_analysis.py
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
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import math
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model_metric import TreeReIdModel


class ImageDataset(Dataset):
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


def analyze():
    model_path = r'E:\tree_id_2.0\models\metric\best_model.pth'
    data_dir = Path(r'E:\tree_id_2.0\data')
    image_base = Path(r'E:\tree_id_2.0\images')
    input_file = 'training_data_cleaned.xlsx'
    batch_size = 64
    val_split = 0.15
    n_prototypes = 3
    outlier_threshold = 0.5
    seed = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)

    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees")

    # Same split as evaluate_metric.py
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

    train_set = set(train_indices)

    # Load model
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

    # Extract embeddings
    print("Extracting embeddings...")
    dataset = ImageDataset(df['image_path'].tolist(), image_base, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=(device.type == 'cuda'), shuffle=False)

    all_embeddings, valid_mask = [], []
    with torch.no_grad():
        for imgs, idx, valid in tqdm(loader, desc='Extracting'):
            imgs = imgs.to(device)
            emb = model(imgs)
            all_embeddings.append(emb.cpu())
            valid_mask.extend([v.item() for v in valid])

    all_embeddings = torch.cat(all_embeddings, 0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)

    # Build prototypes
    print("Building prototypes...")
    tree_prototypes = {}
    tree_train_count = {}
    for key, group in df.groupby('key'):
        indices = [i for i in group.index.tolist()
                   if i in train_set and i < len(valid_mask) and valid_mask[i]]
        if not indices:
            continue
        tree_train_count[key] = len(indices)
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

    # Address mapping
    address_trees = defaultdict(list)
    for key in tree_prototypes:
        addr = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[addr].append(key)

    # GPS coords
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

    # Evaluate with detailed tracking
    print(f"Evaluating {len(test_indices)} held-out photos...")
    errors = []
    correct_details = []
    trivial = 0
    total_nontrivial = 0

    # Per-address stats
    addr_correct = defaultdict(int)
    addr_total = defaultdict(int)

    # Per-tree stats
    tree_correct = defaultdict(int)
    tree_total = defaultdict(int)

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
            continue

        total_nontrivial += 1
        sims = []
        for ck in candidates:
            if ck in tree_prototypes:
                best = max(float(np.dot(query, p)) for p in tree_prototypes[ck])
            else:
                best = 0.0
            sims.append((ck, best))
        sims.sort(key=lambda x: x[1], reverse=True)

        pred = sims[0][0]
        addr_total[addr] += 1
        tree_total[true_key] += 1

        if pred == true_key:
            addr_correct[addr] += 1
            tree_correct[true_key] += 1
            correct_details.append({
                'true_key': true_key, 'addr': addr,
                'margin': sims[0][1] - sims[1][1] if len(sims) > 1 else 1.0,
                'candidates': len(candidates),
                'train_photos': tree_train_count.get(true_key, 0),
            })
        else:
            true_sim = next(s for k, s in sims if k == true_key)
            true_rank = next(i for i, (k, _) in enumerate(sims) if k == true_key) + 1
            errors.append({
                'true_key': true_key, 'pred_key': pred, 'addr': addr,
                'margin': sims[0][1] - true_sim,
                'true_sim': true_sim, 'pred_sim': sims[0][1],
                'true_rank': true_rank,
                'candidates': len(candidates),
                'train_photos': tree_train_count.get(true_key, 0),
                'pred_train_photos': tree_train_count.get(pred, 0),
            })

    # ========== ANALYSIS ==========
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"Total non-trivial: {total_nontrivial}, Correct: {total_nontrivial - len(errors)}, "
          f"Errors: {len(errors)} ({len(errors)/total_nontrivial*100:.1f}%)")
    print(f"Trivial (single-tree address): {trivial}")

    # 1. Margin distribution
    print(f"\n--- MARGIN DISTRIBUTION (errors) ---")
    margins = [e['margin'] for e in errors]
    brackets = [(0, 0.001), (0.001, 0.005), (0.005, 0.01), (0.01, 0.02),
                (0.02, 0.05), (0.05, 0.1), (0.1, 1.0)]
    for lo, hi in brackets:
        count = sum(1 for m in margins if lo <= m < hi)
        print(f"  [{lo:.3f}, {hi:.3f}): {count} ({count/len(errors)*100:.1f}%)")

    # 2. True rank distribution
    print(f"\n--- TRUE RANK IN ERRORS ---")
    ranks = [e['true_rank'] for e in errors]
    rank_counts = Counter(ranks)
    for r in sorted(rank_counts.keys())[:10]:
        print(f"  Rank {r}: {rank_counts[r]} ({rank_counts[r]/len(errors)*100:.1f}%)")

    # 3. Worst addresses
    print(f"\n--- WORST ADDRESSES (by error count) ---")
    addr_errors = defaultdict(int)
    for e in errors:
        addr_errors[e['addr']] += 1
    worst = sorted(addr_errors.items(), key=lambda x: x[1], reverse=True)[:20]
    for addr, err_count in worst:
        total = addr_total[addr]
        correct = addr_correct[addr]
        n_trees = len(address_trees.get(addr, []))
        print(f"  {addr}: {err_count} errors / {total} tests "
              f"({correct/total*100:.0f}% acc), {n_trees} trees at address")

    # 4. Candidate count vs accuracy
    print(f"\n--- CANDIDATE COUNT VS ACCURACY ---")
    cand_correct = defaultdict(int)
    cand_total = defaultdict(int)
    for e in errors:
        cand_total[e['candidates']] += 1
    for c in correct_details:
        cand_correct[c['candidates']] += 1
        cand_total[c['candidates']] += 0  # ensure key exists
    for n_cand in sorted(set(list(cand_total.keys()) + list(cand_correct.keys()))):
        t = cand_total.get(n_cand, 0) + cand_correct.get(n_cand, 0)
        c = cand_correct.get(n_cand, 0)
        if t > 0:
            print(f"  {n_cand:3d} candidates: {c}/{t} correct ({c/t*100:.1f}%)")

    # 5. Training photos vs accuracy
    print(f"\n--- TRAIN PHOTO COUNT VS ACCURACY ---")
    photo_bins = [(1, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 100)]
    for lo, hi in photo_bins:
        err = sum(1 for e in errors if lo <= e['train_photos'] < hi)
        cor = sum(1 for c in correct_details if lo <= c['train_photos'] < hi)
        t = err + cor
        if t > 0:
            print(f"  [{lo:2d}, {hi:2d}) train photos: {cor}/{t} correct ({cor/t*100:.1f}%)")

    # 6. Trees that ALWAYS fail
    print(f"\n--- TREES THAT ALWAYS FAIL (>=2 test photos, 0% accuracy) ---")
    always_fail = []
    for key in tree_total:
        if tree_total[key] >= 2 and tree_correct[key] == 0:
            always_fail.append((key, tree_total[key], tree_train_count.get(key, 0)))
    always_fail.sort(key=lambda x: x[1], reverse=True)
    print(f"  Count: {len(always_fail)} trees always fail")
    for key, n_test, n_train in always_fail[:15]:
        addr = key.rsplit('|', 1)[0]
        n_cand = len(address_trees.get(addr, []))
        print(f"    {key}: {n_test} test photos, {n_train} train photos, {n_cand} candidates at addr")

    # 7. Confusion pairs (most common pred_key when true_key fails)
    print(f"\n--- TOP CONFUSION PAIRS ---")
    pair_counts = Counter((e['true_key'], e['pred_key']) for e in errors)
    for (true, pred), count in pair_counts.most_common(15):
        addr = true.rsplit('|', 1)[0]
        print(f"  {true} → {pred}: {count} times (addr: {addr})")

    # 8. Summary stats
    print(f"\n--- SUMMARY ---")
    # How many addresses are "perfect" (100% accuracy)?
    perfect_addrs = sum(1 for a in addr_total if addr_correct[a] == addr_total[a])
    print(f"  Perfect addresses: {perfect_addrs}/{len(addr_total)}")
    print(f"  Addresses with errors: {len(addr_errors)}")
    # Top 10 worst addresses account for what % of errors?
    top10_errors = sum(c for _, c in worst[:10])
    print(f"  Top 10 worst addresses: {top10_errors}/{len(errors)} errors ({top10_errors/len(errors)*100:.1f}%)")
    top20_errors = sum(c for _, c in worst[:20])
    print(f"  Top 20 worst addresses: {top20_errors}/{len(errors)} errors ({top20_errors/len(errors)*100:.1f}%)")


if __name__ == '__main__':
    analyze()
