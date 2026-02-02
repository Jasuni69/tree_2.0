"""
Detailed failure analysis for tree re-identification holdout evaluation.
Reproduces evaluation with full error tracking and pattern analysis.
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
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


def analyze_failures(model_path: str, data_dir: str = 'data',
                     image_base: str = 'images',
                     input_file: str = 'training_data_cleaned.xlsx',
                     batch_size: int = 64, val_split: float = 0.15,
                     n_prototypes: int = 3, outlier_threshold: float = 0.5,
                     radius_m: float = 30, seed: int = 42):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    np.random.seed(seed)

    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load data
    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees\n")

    # Stratified split (same as eval script)
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

    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    train_set = set(train_indices)

    # Count photos per tree in training set
    train_photos_per_tree = {}
    for key, group in df.groupby('key'):
        train_count = sum(1 for i in group.index if i in train_set)
        train_photos_per_tree[key] = train_count

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
    model.load_state_dict(checkpoint['model_state_dict'])
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
        for imgs, idx, valid in tqdm(loader, desc='Extracting'):
            imgs = imgs.to(device)
            emb = model(imgs)
            all_embeddings.append(emb.cpu())
            valid_mask.extend([v.item() for v in valid])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)
    print(f"Extracted {valid_mask.sum()} valid embeddings\n")

    # Build prototypes from train set
    print("Building prototypes...")
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

    print(f"Trees with prototypes: {len(tree_prototypes)}\n")

    # Address mapping
    address_trees = defaultdict(list)
    for key in tree_prototypes:
        addr = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[addr].append(key)

    # Evaluate with detailed error tracking
    print("Evaluating holdout set...\n")
    results = {k: {'correct': 0, 'total': 0} for k in
               ['proto_only', 'proto_top3']}

    errors = []
    correct_preds = []
    trivial = 0

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

        # Compute similarities to all candidates
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

        # Get true key similarity
        true_sim = next((s for k, s in sims if k == true_key), 0.0)
        pred_sim = sims[0][1]
        margin = pred_sim - true_sim

        # Get addresses
        pred_addr = pred.rsplit('|', 1)[0] if '|' in pred else pred
        true_addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key

        record = {
            'true_key': true_key,
            'pred_key': pred,
            'margin': margin,
            'n_candidates': len(candidates),
            'true_sim': true_sim,
            'pred_sim': pred_sim,
            'address': addr,
            'train_photos_true': train_photos_per_tree.get(true_key, 0),
            'train_photos_pred': train_photos_per_tree.get(pred, 0),
            'same_address': pred_addr == true_addr,
        }

        if pred == true_key:
            results['proto_only']['correct'] += 1
            correct_preds.append(record)
        else:
            errors.append(record)

        if true_key in top3:
            results['proto_top3']['correct'] += 1

    # Print basic results
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test photos: {len(test_indices)}, Trivial cases: {trivial}\n")
    for method, d in results.items():
        if d['total'] > 0:
            acc = d['correct'] / d['total'] * 100
            print(f"  {method:20s}: {acc:5.1f}% ({d['correct']}/{d['total']})")

    # Detailed error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    print(f"\nTotal errors: {len(errors)}")

    if not errors:
        print("No errors found!")
        return

    error_df = pd.DataFrame(errors)

    # Margin analysis
    margins = error_df['margin'].values
    print(f"\nMargin statistics:")
    print(f"  Mean: {margins.mean():.4f}")
    print(f"  Median: {np.median(margins):.4f}")
    print(f"  Min: {margins.min():.4f}, Max: {margins.max():.4f}")
    print(f"  Margin < 0.05: {(margins < 0.05).sum()} ({(margins < 0.05).sum()/len(margins)*100:.1f}%)")
    print(f"  Margin < 0.01: {(margins < 0.01).sum()} ({(margins < 0.01).sum()/len(margins)*100:.1f}%)")

    # Same address errors
    same_addr_errors = error_df['same_address'].sum()
    print(f"\nSame-address errors: {same_addr_errors}/{len(errors)} ({same_addr_errors/len(errors)*100:.1f}%)")

    # Address-level error analysis
    print(f"\n" + "-" * 70)
    print("TOP 20 ADDRESSES WITH MOST ERRORS")
    print("-" * 70)
    addr_errors = error_df['address'].value_counts().head(20)
    for addr, count in addr_errors.items():
        # Get total test cases for this address
        total_at_addr = sum(1 for r in errors + correct_preds if r['address'] == addr)
        pct = count / total_at_addr * 100 if total_at_addr > 0 else 0
        print(f"  {addr[:50]:50s}: {count:3d} errors / {total_at_addr:3d} tests ({pct:.1f}%)")

    # Training photo analysis
    print(f"\n" + "-" * 70)
    print("TRAINING PHOTOS vs ERROR RATE")
    print("-" * 70)

    # Bin by training photo count
    bins = [0, 2, 5, 10, 20, 100]
    bin_labels = ['1-2', '3-5', '6-10', '11-20', '20+']

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        err_in_bin = error_df[(error_df['train_photos_true'] > low) &
                               (error_df['train_photos_true'] <= high)]
        correct_in_bin = [r for r in correct_preds
                          if low < r['train_photos_true'] <= high]

        total_in_bin = len(err_in_bin) + len(correct_in_bin)
        if total_in_bin > 0:
            err_rate = len(err_in_bin) / total_in_bin * 100
            print(f"  {bin_labels[i]:6s} train photos: {len(err_in_bin):3d} errors / {total_in_bin:4d} tests = {err_rate:5.1f}%")

    # Candidate count analysis
    print(f"\n" + "-" * 70)
    print("CANDIDATE COUNT DISTRIBUTION")
    print("-" * 70)

    error_cands = error_df['n_candidates'].values
    correct_cands = [r['n_candidates'] for r in correct_preds]

    print(f"Errors:")
    print(f"  Mean candidates: {error_cands.mean():.1f}")
    print(f"  Median candidates: {np.median(error_cands):.0f}")
    print(f"  Min: {error_cands.min()}, Max: {error_cands.max()}")

    print(f"\nCorrect predictions:")
    print(f"  Mean candidates: {np.mean(correct_cands):.1f}")
    print(f"  Median candidates: {np.median(correct_cands):.0f}")
    print(f"  Min: {min(correct_cands)}, Max: {max(correct_cands)}")

    # Candidate count bins
    print(f"\nError rate by candidate count:")
    cand_bins = [(2, 3), (4, 5), (6, 10), (11, 20), (21, 100)]
    for low, high in cand_bins:
        err_in_bin = error_df[(error_df['n_candidates'] >= low) &
                               (error_df['n_candidates'] <= high)]
        correct_in_bin = [r for r in correct_preds
                          if low <= r['n_candidates'] <= high]
        total_in_bin = len(err_in_bin) + len(correct_in_bin)
        if total_in_bin > 0:
            err_rate = len(err_in_bin) / total_in_bin * 100
            print(f"  {low:2d}-{high:2d} candidates: {len(err_in_bin):3d} errors / {total_in_bin:4d} tests = {err_rate:5.1f}%")

    # Confused tree pairs
    print(f"\n" + "-" * 70)
    print("TOP 20 MOST CONFUSED TREE PAIRS")
    print("-" * 70)

    tree_pairs = [(r['true_key'], r['pred_key']) for r in errors]
    pair_counts = Counter(tree_pairs).most_common(20)

    for (true_key, pred_key), count in pair_counts:
        # Check if confusion is bidirectional
        reverse_count = sum(1 for t, p in tree_pairs if t == pred_key and p == true_key)
        bidirectional = " (bidirectional)" if reverse_count > 0 else ""

        true_short = true_key.split('|')[-1] if '|' in true_key else true_key
        pred_short = pred_key.split('|')[-1] if '|' in pred_key else pred_key
        addr_short = true_key.rsplit('|', 1)[0] if '|' in true_key else "?"

        print(f"  {count:2d}x: {true_short:15s} -> {pred_short:15s} @ {addr_short[:30]:30s}{bidirectional}")

    # Similarity distribution
    print(f"\n" + "-" * 70)
    print("SIMILARITY DISTRIBUTIONS")
    print("-" * 70)

    print(f"\nErrors - similarity to TRUE tree:")
    true_sims_err = error_df['true_sim'].values
    print(f"  Mean: {true_sims_err.mean():.4f}, Median: {np.median(true_sims_err):.4f}")
    print(f"  Min: {true_sims_err.min():.4f}, Max: {true_sims_err.max():.4f}")

    print(f"\nErrors - similarity to PREDICTED tree:")
    pred_sims_err = error_df['pred_sim'].values
    print(f"  Mean: {pred_sims_err.mean():.4f}, Median: {np.median(pred_sims_err):.4f}")
    print(f"  Min: {pred_sims_err.min():.4f}, Max: {pred_sims_err.max():.4f}")

    correct_df = pd.DataFrame(correct_preds)
    print(f"\nCorrect predictions - similarity to TRUE tree:")
    true_sims_correct = correct_df['pred_sim'].values  # pred_sim is similarity to predicted (which is correct)
    print(f"  Mean: {true_sims_correct.mean():.4f}, Median: {np.median(true_sims_correct):.4f}")
    print(f"  Min: {true_sims_correct.min():.4f}, Max: {true_sims_correct.max():.4f}")

    # Save detailed error log
    print(f"\n" + "=" * 70)
    output_file = Path(__file__).parent / 'error_details.csv'
    error_df.to_csv(output_file, index=False)
    print(f"Saved detailed error log to: {output_file}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    analyze_failures(
        model_path='models/metric/best_model.pth',
        data_dir='data',
        image_base='images'
    )
