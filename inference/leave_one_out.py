"""
Leave-one-out evaluation for prototype-based tree identification.

For each photo:
1. Remove it from its tree's embeddings
2. Recompute that tree's mean embedding without it
3. Check if the photo still matches its tree vs other candidates

Runs on CPU to avoid interfering with GPU training.

Run: python inference/leave_one_out.py
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
import timm
import torch.nn as nn
from collections import defaultdict


class TreeClassifier(nn.Module):
    def __init__(self, num_classes, backbone='efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_features(self, x):
        return self.backbone(x)


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
            return torch.zeros(3, 260, 260), idx, False


def leave_one_out(model_path, data_dir, image_base, input_file='training_data_cleaned.xlsx',
                  batch_size=32):
    """
    Leave-one-out evaluation using mean embeddings.
    Runs entirely on CPU.
    """
    device = torch.device('cpu')
    print("Running on CPU (safe alongside GPU training)")

    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load data
    with open(data_dir / 'label_encoder_gt.json') as f:
        encoder = json.load(f)
    num_classes = len(encoder)

    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded {len(df)} photos, {df['key'].nunique()} trees")

    # Load model on CPU
    print("Loading model on CPU...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    backbone = config.get('backbone', 'efficientnet_b2')

    model = TreeClassifier(num_classes, backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 1: Extract all per-photo embeddings on CPU
    print(f"\nExtracting embeddings (CPU, batch_size={batch_size})...")
    all_paths = df['image_path'].tolist()
    dataset = ImageDataset(all_paths, image_base, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                       shuffle=False)

    all_embeddings = []
    valid_mask = []

    with torch.no_grad():
        for batch_imgs, batch_idx, batch_valid in tqdm(loader, desc='Extracting (CPU)'):
            features = model.extract_features(batch_imgs)
            features = F.normalize(features, p=2, dim=1)
            all_embeddings.append(features)
            valid_mask.extend([v.item() for v in batch_valid])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)
    print(f"Extracted {valid_mask.sum()} valid embeddings out of {len(valid_mask)}")

    # Step 2: Group embeddings by tree
    print("\nGrouping by tree...")
    tree_photo_indices = defaultdict(list)  # key -> list of df indices
    for idx, row in df.iterrows():
        if idx < len(valid_mask) and valid_mask[idx]:
            tree_photo_indices[row['key']].append(idx)

    # Precompute tree embedding sums and counts (for fast LOO)
    tree_emb_sums = {}
    tree_emb_counts = {}
    for key, indices in tree_photo_indices.items():
        embs = all_embeddings[indices]
        tree_emb_sums[key] = embs.sum(axis=0)
        tree_emb_counts[key] = len(indices)

    # Build address -> trees mapping
    address_trees = defaultdict(list)
    for key in tree_photo_indices:
        address = key.rsplit('|', 1)[0] if '|' in key else key
        address_trees[address].append(key)

    # Step 3: Leave-one-out evaluation
    print("\nRunning leave-one-out evaluation...")
    correct = 0
    total = 0
    correct_top3 = 0
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='LOO eval'):
        if idx >= len(valid_mask) or not valid_mask[idx]:
            continue

        true_key = row['key']
        query_emb = all_embeddings[idx]
        address = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key

        # Get candidate trees at same address
        candidate_keys = address_trees.get(address, [])
        if len(candidate_keys) < 2:
            # Only one tree at this address - trivially correct
            correct += 1
            correct_top3 += 1
            total += 1
            continue

        # Compute similarity to each candidate
        similarities = []
        for cand_key in candidate_keys:
            if cand_key == true_key:
                # Leave-one-out: remove this photo from the tree's mean
                count = tree_emb_counts[cand_key]
                if count <= 1:
                    # Only photo for this tree - can't do LOO
                    sim = 0.0
                else:
                    loo_mean = (tree_emb_sums[cand_key] - query_emb) / (count - 1)
                    loo_mean = loo_mean / np.linalg.norm(loo_mean)
                    sim = float(np.dot(query_emb, loo_mean))
            else:
                # Normal mean for other trees
                mean = tree_emb_sums[cand_key] / tree_emb_counts[cand_key]
                mean = mean / np.linalg.norm(mean)
                sim = float(np.dot(query_emb, mean))

            similarities.append((cand_key, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        pred_key = similarities[0][0]
        top3_keys = [s[0] for s in similarities[:3]]

        if pred_key == true_key:
            correct += 1
        else:
            errors.append({
                'image_path': row['image_path'],
                'true_key': true_key,
                'pred_key': pred_key,
                'true_sim': next(s for k, s in similarities if k == true_key),
                'pred_sim': similarities[0][1],
                'num_candidates': len(candidate_keys)
            })

        if true_key in top3_keys:
            correct_top3 += 1

        total += 1

    # Results
    accuracy = correct / total * 100
    top3_accuracy = correct_top3 / total * 100

    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT RESULTS (mean embeddings)")
    print("=" * 60)
    print(f"Top-1 Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"Top-3 Accuracy: {top3_accuracy:.1f}% ({correct_top3}/{total})")
    print(f"Errors: {len(errors)}")

    # Stats on single-photo trees
    single_photo_trees = sum(1 for c in tree_emb_counts.values() if c == 1)
    print(f"\nSingle-photo trees: {single_photo_trees} (LOO less meaningful for these)")

    # Error analysis
    if errors:
        print(f"\nError breakdown:")
        # Group by number of candidates
        cand_counts = defaultdict(int)
        for e in errors:
            cand_counts[e['num_candidates']] += 1
        for nc, count in sorted(cand_counts.items()):
            print(f"  {nc} candidates at address: {count} errors")

        # Save errors
        errors_df = pd.DataFrame(errors)
        errors_df = errors_df.sort_values('pred_sim', ascending=False)
        errors_path = data_dir / 'loo_errors.xlsx'
        errors_df.to_excel(errors_path, index=False)
        print(f"\nError details saved to {errors_path}")

    return accuracy, top3_accuracy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\model_seed42.pt')
    parser.add_argument('--data', default=r'E:\tree_id_2.0\data')
    parser.add_argument('--images', default=r'E:\tree_id_2.0\images')
    parser.add_argument('--input', default='training_data_cleaned.xlsx')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    leave_one_out(args.model, args.data, args.images, args.input, args.batch_size)
