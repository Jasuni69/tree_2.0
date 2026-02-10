"""
Audit historical labels using ConvNeXt + DINOv2 ensemble.

Runs model on all photos, compares predictions to current labels.
Outputs report of suspected mislabels with confidence scores.

Run: python training/audit_labels.py
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
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from model_metric import TreeReIdModel


class AuditDataset(Dataset):
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


def build_prototypes(df, emb, valid, n_prototypes=3, outlier_threshold=0.5):
    """Build prototypes from ALL photos (not train/test split)."""
    tree_prototypes = {}
    tree_photo_indices = defaultdict(list)

    for idx, row in df.iterrows():
        if idx < len(valid) and valid[idx]:
            tree_photo_indices[row['key']].append(idx)

    for key, indices in tree_photo_indices.items():
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


def audit():
    # Paths
    CONVNEXT_MODEL = r'E:\tree_id_2.0\models\metric_384\best_model.pth'
    DINOV2_MODEL = r'E:\tree_id_2.0\models\metric_dinov2\best_model.pth'
    data_dir = Path(r'E:\tree_id_2.0\data')
    image_base = Path(r'E:\tree_id_2.0\images')
    input_file = 'training_data_cleaned.xlsx'
    output_file = data_dir / f'label_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

    GPS_RADIUS = 15.0
    N_PROTOTYPES = 3
    OUTLIER_THRESHOLD = 0.5
    MISMATCH_THRESHOLD = 0.1  # Flag if predicted similarity is this much higher than current label

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees")

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
    print(f"DINOv2: input={dino_input_size}")

    ds_dino = AuditDataset(image_paths, image_base, input_size=dino_input_size)
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
    print(f"ConvNeXt: input={convnext_input_size}")

    ds_convnext = AuditDataset(image_paths, image_base, input_size=convnext_input_size)
    emb_convnext, valid_convnext = extract_embeddings(convnext, ds_convnext, device, 32, 'ConvNeXt')

    del convnext
    torch.cuda.empty_cache()

    # ========== COMBINE EMBEDDINGS ==========
    valid = valid_dino & valid_convnext
    print(f"\nValid photos: {valid.sum()}/{len(valid)}")

    # L2-normalize and concat
    emb_dino_norm = emb_dino / (np.linalg.norm(emb_dino, axis=1, keepdims=True) + 1e-8)
    emb_convnext_norm = emb_convnext / (np.linalg.norm(emb_convnext, axis=1, keepdims=True) + 1e-8)
    emb_concat = np.concatenate([emb_dino_norm, emb_convnext_norm], axis=1)
    emb_concat = emb_concat / (np.linalg.norm(emb_concat, axis=1, keepdims=True) + 1e-8)

    # ========== BUILD PROTOTYPES ==========
    print("\nBuilding prototypes from all photos...")
    prototypes = build_prototypes(df, emb_concat, valid, N_PROTOTYPES, OUTLIER_THRESHOLD)
    print(f"Built prototypes for {len(prototypes)} trees")

    # GPS coords
    trees = df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
    tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

    # ========== AUDIT EACH PHOTO ==========
    print("\nAuditing labels...")

    audit_results = []
    mismatches = []

    for idx in tqdm(range(len(df)), desc='Auditing'):
        if not valid[idx]:
            continue

        row = df.iloc[idx]
        current_key = row['key']
        query = emb_concat[idx]

        # Get photo GPS
        plat, plon = row['photo_lat'], row['photo_lon']

        # Find all trees within GPS radius
        candidates = []
        for key, (tlat, tlon) in tree_coords.items():
            if key not in prototypes:
                continue
            dist = haversine(plat, plon, tlat, tlon)
            if dist <= GPS_RADIUS:
                candidates.append((key, dist))

        if not candidates:
            continue

        # Score all candidates
        scores = []
        for key, dist in candidates:
            best_sim = max(float(np.dot(query, p)) for p in prototypes[key])
            scores.append({
                'key': key,
                'similarity': best_sim,
                'distance_m': dist
            })

        scores.sort(key=lambda x: x['similarity'], reverse=True)

        # Current label score
        current_score = next((s for s in scores if s['key'] == current_key), None)
        current_sim = current_score['similarity'] if current_score else 0.0

        # Best prediction
        pred = scores[0]
        pred_key = pred['key']
        pred_sim = pred['similarity']

        # Record result
        result = {
            'idx': idx,
            'image_path': row['image_path'],
            'address': row['address'],
            'current_key': current_key,
            'current_sim': current_sim,
            'predicted_key': pred_key,
            'predicted_sim': pred_sim,
            'sim_diff': pred_sim - current_sim,
            'match': current_key == pred_key,
            'n_candidates': len(candidates),
        }
        audit_results.append(result)

        # Flag mismatches
        if current_key != pred_key and (pred_sim - current_sim) > MISMATCH_THRESHOLD:
            result['rank_of_current'] = next((i+1 for i, s in enumerate(scores) if s['key'] == current_key), -1)
            mismatches.append(result)

    # ========== REPORT ==========
    print(f"\n{'='*70}")
    print("AUDIT RESULTS")
    print(f"{'='*70}")

    total = len(audit_results)
    matches = sum(1 for r in audit_results if r['match'])
    print(f"Total photos audited: {total}")
    print(f"Labels match prediction: {matches} ({matches/total*100:.1f}%)")
    print(f"Labels differ from prediction: {total - matches} ({(total-matches)/total*100:.1f}%)")
    print(f"High-confidence mismatches: {len(mismatches)}")

    # Save full results
    results_df = pd.DataFrame(audit_results)
    results_df.to_excel(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")

    # Save mismatches separately
    if mismatches:
        mismatch_file = output_file.with_name(output_file.stem + '_mismatches.xlsx')
        mismatch_df = pd.DataFrame(mismatches)
        mismatch_df = mismatch_df.sort_values('sim_diff', ascending=False)
        mismatch_df.to_excel(mismatch_file, index=False)
        print(f"Mismatches saved to: {mismatch_file}")

        print(f"\nTop 20 suspected mislabels:")
        print("-" * 70)
        for i, row in mismatch_df.head(20).iterrows():
            print(f"  {row['image_path']}")
            print(f"    Current: {row['current_key']} (sim={row['current_sim']:.3f})")
            print(f"    Predicted: {row['predicted_key']} (sim={row['predicted_sim']:.3f}, diff=+{row['sim_diff']:.3f})")
            print()

    print("Done.")


if __name__ == '__main__':
    audit()
