"""
Ensemble evaluation: average embeddings from 224 and 384 models.
Different resolutions see different features — free accuracy boost.

Run: python training/evaluate_ensemble.py
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


def load_model(model_path, device):
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
    return model, input_size


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


def evaluate():
    MODEL_224 = r'E:\tree_id_2.0\models\metric\best_model.pth'
    MODEL_384 = r'E:\tree_id_2.0\models\metric_384\best_model.pth'
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

    # Load both models
    print(f"\nLoading 224 model...")
    model_224, size_224 = load_model(MODEL_224, device)
    print(f"Loading 384 model...")
    model_384, size_384 = load_model(MODEL_384, device)
    print(f"Sizes: {size_224}, {size_384}")

    image_paths = df['image_path'].tolist()

    # Extract embeddings from both models
    print(f"\nExtracting 224 embeddings...")
    ds_224 = SimpleDataset(image_paths, image_base, input_size=size_224)
    emb_224, valid_224 = extract_embeddings(model_224, ds_224, device, 64, '224')

    # Free GPU memory before loading 384
    del model_224
    torch.cuda.empty_cache()

    print(f"\nExtracting 384 embeddings...")
    ds_384 = SimpleDataset(image_paths, image_base, input_size=size_384)
    emb_384, valid_384 = extract_embeddings(model_384, ds_384, device, 32, '384')

    del model_384
    torch.cuda.empty_cache()

    # Combine: L2-normalize each, average, re-normalize
    valid = valid_224 & valid_384
    emb_224_norm = emb_224 / (np.linalg.norm(emb_224, axis=1, keepdims=True) + 1e-8)
    emb_384_norm = emb_384 / (np.linalg.norm(emb_384, axis=1, keepdims=True) + 1e-8)
    emb_ensemble = emb_224_norm + emb_384_norm
    emb_ensemble = emb_ensemble / (np.linalg.norm(emb_ensemble, axis=1, keepdims=True) + 1e-8)

    print(f"Valid: {valid.sum()} embeddings")

    # Build prototypes from ensemble embeddings
    print("\nBuilding ensemble prototypes...")
    tree_prototypes = {}
    for key, group in df.groupby('key'):
        indices = [i for i in group.index.tolist()
                   if i in train_set and i < len(valid) and valid[i]]
        if not indices:
            continue
        embs = emb_ensemble[indices]
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

    # Also build single-model prototypes for comparison
    proto_224 = {}
    proto_384 = {}
    for key, group in df.groupby('key'):
        indices = [i for i in group.index.tolist()
                   if i in train_set and i < len(valid) and valid[i]]
        if not indices:
            continue
        for emb_src, proto_dict in [(emb_224_norm, proto_224), (emb_384_norm, proto_384)]:
            embs = emb_src[indices]
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
            proto_dict[key] = protos

    # GPS coords + address mapping
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

    address_trees = defaultdict(list)
    for key in tree_prototypes:
        addr = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[addr].append(key)

    # Evaluate all three: 224-only, 384-only, ensemble
    configs = [
        ('224 only', emb_224_norm, proto_224),
        ('384 only', emb_384_norm, proto_384),
        ('ensemble', emb_ensemble, tree_prototypes),
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

        print(f"\n--- {name.upper()} ---")
        print(f"Trivial: {trivial}")
        for method, d in results.items():
            if d['total'] > 0:
                acc = d['correct'] / d['total'] * 100
                print(f"  {method:20s}: {acc:5.1f}% ({d['correct']}/{d['total']})")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == '__main__':
    evaluate()
