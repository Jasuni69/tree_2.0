"""
Enhanced system-level evaluation with GPS tuning and multi-photo voting.
Tests multiple GPS radii, voting strategies, and combined scoring.

Run: python training/evaluate_system.py
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


def get_best_similarity(query, prototypes):
    """Max similarity to any prototype."""
    return max(float(np.dot(query, p)) for p in prototypes)


def gps_radius_sweep(df, test_indices, all_embeddings, valid_mask,
                     tree_prototypes, tree_coords, radii):
    """Test multiple GPS radii."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: GPS RADIUS SWEEP")
    print("=" * 60)

    results = {}
    for radius in radii:
        correct, total = 0, 0
        gps_found = 0

        for idx in test_indices:
            if idx >= len(valid_mask) or not valid_mask[idx]:
                continue
            row = df.iloc[idx]
            true_key = row['key']
            if true_key not in tree_prototypes:
                continue

            query = all_embeddings[idx]
            plat, plon = row['photo_lat'], row['photo_lon']

            # Find GPS candidates
            gps_cands = []
            for k, (tlat, tlon) in tree_coords.items():
                if k not in tree_prototypes:
                    continue
                d = haversine(plat, plon, tlat, tlon)
                if d <= radius:
                    sim = get_best_similarity(query, tree_prototypes[k])
                    gps_cands.append((k, sim))

            if gps_cands:
                gps_found += 1
                gps_cands.sort(key=lambda x: x[1], reverse=True)
                pred = gps_cands[0][0]
                total += 1
                if pred == true_key:
                    correct += 1

        coverage = gps_found / len([i for i in test_indices
                                     if i < len(valid_mask) and valid_mask[i]]) * 100
        acc = correct / total * 100 if total > 0 else 0
        results[radius] = (acc, correct, total, coverage)

    print("\nRadius (m) | Accuracy | Correct/Total | Coverage")
    print("-" * 60)
    for radius in radii:
        acc, correct, total, cov = results[radius]
        print(f"{radius:10.0f} | {acc:7.1f}% | {correct:6d}/{total:5d} | {cov:6.1f}%")

    return results


def multi_photo_voting(df, test_indices, all_embeddings, valid_mask,
                       tree_prototypes, tree_coords, radius_m=30):
    """Multi-photo voting strategy."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: MULTI-PHOTO VOTING")
    print("=" * 60)

    # Group test photos by tree
    tree_test_photos = defaultdict(list)
    for idx in test_indices:
        if idx >= len(valid_mask) or not valid_mask[idx]:
            continue
        row = df.iloc[idx]
        true_key = row['key']
        if true_key in tree_prototypes:
            tree_test_photos[true_key].append(idx)

    # Single photo accuracy
    single_correct, single_total = 0, 0

    # Multi-photo voting accuracy
    voting_correct, voting_total = 0, 0

    # Embedding fusion accuracy
    fusion_correct, fusion_total = 0, 0

    for true_key, photo_indices in tree_test_photos.items():
        addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key

        # Single photo baseline
        for idx in photo_indices:
            row = df.iloc[idx]
            query = all_embeddings[idx]
            plat, plon = row['photo_lat'], row['photo_lon']

            # GPS candidates
            gps_cands = []
            for k, (tlat, tlon) in tree_coords.items():
                if k not in tree_prototypes:
                    continue
                d = haversine(plat, plon, tlat, tlon)
                if d <= radius_m:
                    sim = get_best_similarity(query, tree_prototypes[k])
                    gps_cands.append((k, sim))

            if gps_cands:
                gps_cands.sort(key=lambda x: x[1], reverse=True)
                pred = gps_cands[0][0]
                single_total += 1
                if pred == true_key:
                    single_correct += 1

        # Multi-photo voting (only if 2+ photos)
        if len(photo_indices) >= 2:
            votes = []
            for idx in photo_indices:
                row = df.iloc[idx]
                query = all_embeddings[idx]
                plat, plon = row['photo_lat'], row['photo_lon']

                gps_cands = []
                for k, (tlat, tlon) in tree_coords.items():
                    if k not in tree_prototypes:
                        continue
                    d = haversine(plat, plon, tlat, tlon)
                    if d <= radius_m:
                        sim = get_best_similarity(query, tree_prototypes[k])
                        gps_cands.append((k, sim))

                if gps_cands:
                    gps_cands.sort(key=lambda x: x[1], reverse=True)
                    votes.append(gps_cands[0][0])

            if votes:
                most_common = Counter(votes).most_common(1)[0][0]
                voting_total += 1
                if most_common == true_key:
                    voting_correct += 1

            # Embedding fusion: average embeddings first
            embeddings = [all_embeddings[idx] for idx in photo_indices]
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)

            # Use first photo GPS (could also average GPS)
            first_row = df.iloc[photo_indices[0]]
            plat, plon = first_row['photo_lat'], first_row['photo_lon']

            gps_cands = []
            for k, (tlat, tlon) in tree_coords.items():
                if k not in tree_prototypes:
                    continue
                d = haversine(plat, plon, tlat, tlon)
                if d <= radius_m:
                    sim = get_best_similarity(avg_emb, tree_prototypes[k])
                    gps_cands.append((k, sim))

            if gps_cands:
                gps_cands.sort(key=lambda x: x[1], reverse=True)
                pred = gps_cands[0][0]
                fusion_total += 1
                if pred == true_key:
                    fusion_correct += 1

    single_acc = single_correct / single_total * 100 if single_total > 0 else 0
    voting_acc = voting_correct / voting_total * 100 if voting_total > 0 else 0
    fusion_acc = fusion_correct / fusion_total * 100 if fusion_total > 0 else 0

    print(f"\nSingle-photo accuracy: {single_acc:.1f}% ({single_correct}/{single_total})")
    print(f"Multi-photo voting:    {voting_acc:.1f}% ({voting_correct}/{voting_total}) [trees with 2+ photos]")
    print(f"Embedding fusion:      {fusion_acc:.1f}% ({fusion_correct}/{fusion_total}) [trees with 2+ photos]")
    print(f"\nTrees with 2+ test photos: {voting_total}")

    return {
        'single': (single_acc, single_correct, single_total),
        'voting': (voting_acc, voting_correct, voting_total),
        'fusion': (fusion_acc, fusion_correct, fusion_total)
    }


def combined_scoring(df, test_indices, all_embeddings, valid_mask,
                     tree_prototypes, tree_coords, radius_m=30, alphas=[0.7, 0.8, 0.9, 1.0]):
    """GPS + Embedding reranking."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: COMBINED GPS + EMBEDDING SCORING")
    print("=" * 60)
    print("score = alpha * cosine_sim + (1-alpha) * (1 - dist/radius)")

    results = {}
    for alpha in alphas:
        correct, total = 0, 0

        for idx in test_indices:
            if idx >= len(valid_mask) or not valid_mask[idx]:
                continue
            row = df.iloc[idx]
            true_key = row['key']
            if true_key not in tree_prototypes:
                continue

            query = all_embeddings[idx]
            plat, plon = row['photo_lat'], row['photo_lon']

            # Find GPS candidates with combined score
            gps_cands = []
            for k, (tlat, tlon) in tree_coords.items():
                if k not in tree_prototypes:
                    continue
                d = haversine(plat, plon, tlat, tlon)
                if d <= radius_m:
                    sim = get_best_similarity(query, tree_prototypes[k])
                    # Combined score: closer trees get bonus
                    dist_score = 1.0 - (d / radius_m)
                    combined = alpha * sim + (1.0 - alpha) * dist_score
                    gps_cands.append((k, combined))

            if gps_cands:
                gps_cands.sort(key=lambda x: x[1], reverse=True)
                pred = gps_cands[0][0]
                total += 1
                if pred == true_key:
                    correct += 1

        acc = correct / total * 100 if total > 0 else 0
        results[alpha] = (acc, correct, total)

    print("\nAlpha | Accuracy | Correct/Total")
    print("-" * 60)
    for alpha in alphas:
        acc, correct, total = results[alpha]
        print(f"{alpha:5.2f} | {acc:7.1f}% | {correct}/{total}")
    print("\n(alpha=1.0 = pure embedding similarity)")

    return results


def adaptive_strategy(df, test_indices, all_embeddings, valid_mask,
                      tree_prototypes, tree_coords, address_trees,
                      radius_m=30, margin_threshold=0.1):
    """Adaptive strategy based on candidate count."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: CANDIDATE COUNT ADAPTIVE STRATEGY")
    print("=" * 60)

    # Bucket results
    buckets = {
        'few_candidates': {'correct': 0, 'total': 0},  # <= 3
        'many_candidates_confident': {'correct': 0, 'total': 0},  # > 10, high margin
        'many_candidates_uncertain': {'correct': 0, 'total': 0},  # > 10, low margin
        'mid_candidates': {'correct': 0, 'total': 0}  # 4-10
    }

    for idx in test_indices:
        if idx >= len(valid_mask) or not valid_mask[idx]:
            continue
        row = df.iloc[idx]
        true_key = row['key']
        if true_key not in tree_prototypes:
            continue

        query = all_embeddings[idx]
        plat, plon = row['photo_lat'], row['photo_lon']

        # GPS candidates
        gps_cands = []
        for k, (tlat, tlon) in tree_coords.items():
            if k not in tree_prototypes:
                continue
            d = haversine(plat, plon, tlat, tlon)
            if d <= radius_m:
                sim = get_best_similarity(query, tree_prototypes[k])
                gps_cands.append((k, sim))

        if not gps_cands:
            continue

        gps_cands.sort(key=lambda x: x[1], reverse=True)
        pred = gps_cands[0][0]
        n_cands = len(gps_cands)

        # Calculate margin if 2+ candidates
        margin = 0.0
        if len(gps_cands) >= 2:
            margin = gps_cands[0][1] - gps_cands[1][1]

        # Categorize
        if n_cands <= 3:
            bucket = 'few_candidates'
        elif n_cands > 10:
            if margin >= margin_threshold:
                bucket = 'many_candidates_confident'
            else:
                bucket = 'many_candidates_uncertain'
        else:
            bucket = 'mid_candidates'

        buckets[bucket]['total'] += 1
        if pred == true_key:
            buckets[bucket]['correct'] += 1

    print("\nCategory                      | Accuracy | Correct/Total")
    print("-" * 60)
    for bucket_name, data in buckets.items():
        if data['total'] > 0:
            acc = data['correct'] / data['total'] * 100
            label = bucket_name.replace('_', ' ').title()
            print(f"{label:29s} | {acc:7.1f}% | {data['correct']}/{data['total']}")

    return buckets


def evaluate_system(model_path: str, data_dir: str = r'E:\tree_id_2.0\data',
                   image_base: str = r'E:\tree_id_2.0\images',
                   input_file: str = 'training_data_cleaned.xlsx',
                   batch_size: int = 64, val_split: float = 0.15,
                   n_prototypes: int = 3, outlier_threshold: float = 0.5,
                   seed: int = 42):

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

    # Extract embeddings
    print(f"\nExtracting embeddings...")
    dataset = ImageDataset(df['image_path'].tolist(), image_base, transform)
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

    # Run experiments
    radii = [5, 10, 15, 20, 30, 50]
    gps_results = gps_radius_sweep(df, test_indices, all_embeddings, valid_mask,
                                   tree_prototypes, tree_coords, radii)

    voting_results = multi_photo_voting(df, test_indices, all_embeddings, valid_mask,
                                       tree_prototypes, tree_coords, radius_m=30)

    combined_results = combined_scoring(df, test_indices, all_embeddings, valid_mask,
                                       tree_prototypes, tree_coords, radius_m=30)

    adaptive_results = adaptive_strategy(df, test_indices, all_embeddings, valid_mask,
                                        tree_prototypes, tree_coords, address_trees,
                                        radius_m=30, margin_threshold=0.1)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nKEY FINDINGS:")
    print("-" * 60)

    # Best radius
    best_radius = max(gps_results.keys(), key=lambda r: gps_results[r][0])
    print(f"Best GPS radius: {best_radius}m -> {gps_results[best_radius][0]:.1f}% accuracy")

    # Voting improvement
    single_acc = voting_results['single'][0]
    voting_acc = voting_results['voting'][0]
    if voting_acc > single_acc:
        print(f"Multi-photo voting gain: +{voting_acc - single_acc:.1f}% (from {single_acc:.1f}% to {voting_acc:.1f}%)")

    # Combined scoring
    best_alpha = max(combined_results.keys(), key=lambda a: combined_results[a][0])
    print(f"Best alpha: {best_alpha} -> {combined_results[best_alpha][0]:.1f}% accuracy")

    # Adaptive strategy insight
    conf = adaptive_results['many_candidates_confident']
    unconf = adaptive_results['many_candidates_uncertain']
    if conf['total'] > 0 and unconf['total'] > 0:
        conf_acc = conf['correct'] / conf['total'] * 100
        unconf_acc = unconf['correct'] / unconf['total'] * 100
        print(f"Confident vs uncertain: {conf_acc:.1f}% vs {unconf_acc:.1f}% (margin threshold=0.1)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\metric\best_model.pth')
    args = parser.parse_args()
    evaluate_system(args.model)
