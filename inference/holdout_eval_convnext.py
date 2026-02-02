"""
Holdout evaluation using ConvNeXt re-id model (metric learning).

Same holdout logic as holdout_eval.py but uses the ConvNeXt-Base model
trained with ArcFace + Triplet loss from tree_id_new project.
"""

import sys
sys.path.insert(0, r'E:\tree_id_new\tree_id\src')

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

from tree_reid_model_convnext import TreeReIdModelConvNeXt


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


def holdout_eval_convnext(model_path: str, data_dir: str, image_base: str,
                          input_file: str = 'training_data_cleaned.xlsx',
                          batch_size: int = 64, val_split: float = 0.15,
                          n_prototypes: int = 3, outlier_threshold: float = 0.5,
                          radius_m: float = 30, seed: int = 42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    np.random.seed(seed)

    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load our data
    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees")

    # Stratified split (same as holdout_eval.py)
    train_indices = []
    test_indices = []
    trees_skipped = 0

    for key, group in df.groupby('key'):
        indices = group.index.tolist()
        n = len(indices)
        if n < 2:
            train_indices.extend(indices)
            trees_skipped += 1
            continue
        np.random.shuffle(indices)
        n_test = max(1, int(n * val_split))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    print(f"Train: {len(train_indices)} photos")
    print(f"Test:  {len(test_indices)} photos")
    print(f"Trees with only 1 photo (train only): {trees_skipped}")

    train_set = set(train_indices)

    # Load ConvNeXt model
    print(f"\nLoading ConvNeXt re-id model...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {}).get('model', {})

    model = TreeReIdModelConvNeXt(
        backbone_name=config.get('backbone_name', 'convnext_base'),
        embedding_dim=config.get('embedding_dim', 1024),
        pretrained=False,
        freeze_stages=0,
        dropout_rate=config.get('dropout_rate', 0.3),
        pooling=config.get('pooling', 'gem'),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {config.get('backbone_name', 'convnext_base')}, "
          f"emb_dim={config.get('embedding_dim', 1024)}")

    # Transform (ConvNeXt uses 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract all embeddings
    print(f"\nExtracting embeddings ({device}, batch_size={batch_size})...")
    all_paths = df['image_path'].tolist()
    dataset = ImageDataset(all_paths, image_base, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=(device.type == 'cuda'), shuffle=False)

    all_embeddings = []
    valid_mask = []

    with torch.no_grad():
        for batch_imgs, batch_idx, batch_valid in tqdm(loader, desc=f'Extracting ({device})'):
            batch_imgs = batch_imgs.to(device)
            embeddings = model(batch_imgs)  # Already L2-normalized
            all_embeddings.append(embeddings.cpu())
            valid_mask.extend([v.item() for v in batch_valid])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)
    print(f"Extracted {valid_mask.sum()} valid embeddings (dim={all_embeddings.shape[1]})")

    # Build prototypes from TRAIN photos only
    print(f"\nBuilding prototypes from train set...")
    tree_prototypes = {}
    outliers_removed = 0
    total_train_photos = 0

    for key, group in df.groupby('key'):
        indices = [idx for idx in group.index.tolist()
                   if idx in train_set and idx < len(valid_mask) and valid_mask[idx]]
        if not indices:
            continue

        tree_embs = all_embeddings[indices]
        total_train_photos += len(tree_embs)

        # Outlier filtering
        mean_emb = tree_embs.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        similarities = tree_embs @ mean_emb
        keep_mask = similarities >= outlier_threshold
        n_removed = (~keep_mask).sum()
        outliers_removed += n_removed

        if keep_mask.sum() == 0:
            keep_mask[similarities.argmax()] = True

        filtered_embs = tree_embs[keep_mask]

        # K-means prototypes
        n_samples = len(filtered_embs)
        actual_prototypes = min(n_prototypes, n_samples)

        if actual_prototypes > 1 and n_samples >= 3:
            kmeans = KMeans(n_clusters=actual_prototypes, random_state=42, n_init=10)
            kmeans.fit(filtered_embs)
            prototypes = []
            for center in kmeans.cluster_centers_:
                center = center / np.linalg.norm(center)
                prototypes.append(center)
            tree_prototypes[key] = prototypes
        else:
            final_mean = filtered_embs.mean(axis=0)
            final_mean = final_mean / np.linalg.norm(final_mean)
            tree_prototypes[key] = [final_mean]

    print(f"Outliers removed: {outliers_removed}/{total_train_photos}")
    print(f"Trees with prototypes: {len(tree_prototypes)}")

    # Tree coordinates for GPS
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {row['key']: (row['gt_lat'], row['gt_lon']) for _, row in trees.iterrows()}

    # Address -> trees mapping
    address_trees = defaultdict(list)
    for key in tree_prototypes:
        address = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[address].append(key)

    # Evaluate test photos
    print(f"\nEvaluating {len(test_indices)} held-out photos...")

    results = {
        'proto_only': {'correct': 0, 'total': 0},
        'proto_top3': {'correct': 0, 'total': 0},
        'gps_proto': {'correct': 0, 'total': 0},
        'gps_proto_top3': {'correct': 0, 'total': 0},
    }
    trivial_correct = 0
    errors = []

    for idx in tqdm(test_indices, desc='Evaluating holdout'):
        if idx >= len(valid_mask) or not valid_mask[idx]:
            continue

        row = df.iloc[idx]
        true_key = row['key']
        query_emb = all_embeddings[idx]

        if true_key not in tree_prototypes:
            continue

        # Address-based matching
        address = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key
        candidate_keys = address_trees.get(address, [])

        if len(candidate_keys) < 2:
            trivial_correct += 1
            for method in results:
                results[method]['correct'] += 1
                results[method]['total'] += 1
            continue

        similarities = []
        for cand_key in candidate_keys:
            if cand_key in tree_prototypes:
                proto_sims = [float(np.dot(query_emb, p)) for p in tree_prototypes[cand_key]]
                best_sim = max(proto_sims)
            else:
                best_sim = 0.0
            similarities.append((cand_key, best_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        pred_key = similarities[0][0]
        top3_keys = [s[0] for s in similarities[:3]]

        results['proto_only']['total'] += 1
        results['proto_top3']['total'] += 1
        if pred_key == true_key:
            results['proto_only']['correct'] += 1
        else:
            true_sim = next(s for k, s in similarities if k == true_key)
            errors.append({
                'image_path': row['image_path'],
                'true_key': true_key,
                'pred_key': pred_key,
                'true_sim': true_sim,
                'pred_sim': similarities[0][1],
                'num_candidates': len(candidate_keys),
                'margin': similarities[0][1] - true_sim
            })
        if true_key in top3_keys:
            results['proto_top3']['correct'] += 1

        # GPS radius matching
        photo_lat, photo_lon = row['photo_lat'], row['photo_lon']
        gps_candidates = []
        for key, (t_lat, t_lon) in tree_coords.items():
            if key not in tree_prototypes:
                continue
            dist = haversine(photo_lat, photo_lon, t_lat, t_lon)
            if dist <= radius_m:
                proto_sims = [float(np.dot(query_emb, p)) for p in tree_prototypes[key]]
                gps_candidates.append((key, max(proto_sims), dist))

        if gps_candidates:
            gps_candidates.sort(key=lambda x: x[1], reverse=True)
            gps_pred = gps_candidates[0][0]
            gps_top3 = [c[0] for c in gps_candidates[:3]]

            results['gps_proto']['total'] += 1
            results['gps_proto_top3']['total'] += 1
            if gps_pred == true_key:
                results['gps_proto']['correct'] += 1
            if true_key in gps_top3:
                results['gps_proto_top3']['correct'] += 1

    # Results
    print("\n" + "=" * 60)
    print("HOLDOUT RESULTS - ConvNeXt Re-ID (ArcFace + Triplet)")
    print("=" * 60)
    print(f"Test photos: {len(test_indices)}")
    print(f"Trivially correct (single tree at address): {trivial_correct}")
    print()

    for method, data in results.items():
        if data['total'] > 0:
            acc = data['correct'] / data['total'] * 100
            print(f"  {method:20s}: {acc:5.1f}% ({data['correct']}/{data['total']})")

    # Non-trivial
    print(f"\n--- Non-trivial only (multi-tree addresses) ---")
    for method in ['proto_only', 'proto_top3']:
        d = results[method]
        nt_correct = d['correct'] - trivial_correct
        nt_total = d['total'] - trivial_correct
        if nt_total > 0:
            acc = nt_correct / nt_total * 100
            print(f"  {method:20s}: {acc:5.1f}% ({nt_correct}/{nt_total})")

    if errors:
        print(f"\nError analysis ({len(errors)} errors):")
        margins = [e['margin'] for e in errors]
        print(f"  Avg margin: {np.mean(margins):.4f}")
        print(f"  Close calls (margin < 0.05): {sum(1 for m in margins if m < 0.05)}")

        errors_df = pd.DataFrame(errors).sort_values('margin')
        errors_path = data_dir / 'holdout_errors_convnext.xlsx'
        errors_df.to_excel(errors_path, index=False)
        print(f"  Errors saved to {errors_path}")

    return results


if __name__ == '__main__':
    holdout_eval_convnext(
        model_path=r'E:\tree_id_new\tree_id\output\models\tree_reid_convnext\best_model.pth',
        data_dir=r'E:\tree_id_2.0\data',
        image_base=r'E:\tree_id_2.0\images',
    )
