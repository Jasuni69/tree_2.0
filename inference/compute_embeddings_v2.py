"""
Compute tree embeddings with:
1. Multiple prototypes per tree (k-means clustering)
2. Outlier filtering (remove embeddings far from mean)

Run: python compute_embeddings_v2.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from torchvision import transforms
import timm
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


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


def compute_embeddings_v2(model_path, data_dir, image_base, output_path,
                          batch_size=64, n_prototypes=3, outlier_threshold=0.5,
                          input_file='training_data_with_ground_truth.xlsx'):
    """
    Compute embeddings with multiple prototypes and outlier filtering.

    Args:
        n_prototypes: Number of cluster centers per tree (if enough photos)
        outlier_threshold: Remove embeddings with similarity < threshold to mean
        input_file: Input data file name
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Prototypes per tree: {n_prototypes}")
    print(f"Outlier threshold: {outlier_threshold}")
    print(f"Input file: {input_file}")

    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load data
    with open(data_dir / 'label_encoder_gt.json') as f:
        encoder = json.load(f)
    num_classes = len(encoder)

    df = pd.read_excel(data_dir / input_file)
    print(f"Total photos: {len(df)}")
    print(f"Unique trees: {df['key'].nunique()}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    backbone = config.get('backbone', 'efficientnet_b2')

    model = TreeClassifier(num_classes, backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    feature_dim = model.backbone.num_features
    print(f"Feature dim: {feature_dim}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process all images
    all_paths = df['image_path'].tolist()
    dataset = ImageDataset(all_paths, image_base, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                       pin_memory=True, shuffle=False)

    # Extract all embeddings
    all_embeddings = []
    valid_mask = []

    print(f"\nExtracting embeddings...")
    with torch.no_grad():
        for batch_imgs, batch_idx, batch_valid in tqdm(loader, desc='Extracting'):
            batch_imgs = batch_imgs.to(device)
            features = model.extract_features(batch_imgs)
            features = F.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu())
            valid_mask.extend([v.item() for v in batch_valid])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    valid_mask = np.array(valid_mask, dtype=bool)

    # Process each tree
    print(f"\nProcessing trees with clustering and outlier filtering...")

    tree_embeddings = {}  # key -> single average embedding
    tree_prototypes = {}  # key -> list of prototype embeddings
    tree_stats = {}       # key -> stats dict

    outliers_removed = 0
    total_photos = 0

    for key in tqdm(df['key'].unique(), desc='Processing'):
        tree_mask = df['key'] == key
        indices = df[tree_mask].index.tolist()

        # Get valid embeddings for this tree
        tree_embs = []
        for idx in indices:
            if idx < len(valid_mask) and valid_mask[idx]:
                tree_embs.append(all_embeddings[idx])

        if not tree_embs:
            continue

        tree_embs = np.array(tree_embs)
        total_photos += len(tree_embs)

        # Step 1: Compute mean and filter outliers
        mean_emb = tree_embs.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        # Compute similarities to mean
        similarities = tree_embs @ mean_emb

        # Filter outliers
        keep_mask = similarities >= outlier_threshold
        n_removed = (~keep_mask).sum()
        outliers_removed += n_removed

        if keep_mask.sum() == 0:
            # Keep at least one (the best one)
            keep_mask[similarities.argmax()] = True

        filtered_embs = tree_embs[keep_mask]

        # Step 2: Compute final mean from filtered embeddings
        final_mean = filtered_embs.mean(axis=0)
        final_mean = final_mean / np.linalg.norm(final_mean)
        tree_embeddings[key] = torch.from_numpy(final_mean).float()

        # Step 3: Create multiple prototypes via clustering
        n_samples = len(filtered_embs)
        actual_prototypes = min(n_prototypes, n_samples)

        if actual_prototypes > 1 and n_samples >= 3:
            # K-means clustering
            kmeans = KMeans(n_clusters=actual_prototypes, random_state=42, n_init=10)
            kmeans.fit(filtered_embs)

            # Normalize cluster centers
            prototypes = []
            for center in kmeans.cluster_centers_:
                center = center / np.linalg.norm(center)
                prototypes.append(torch.from_numpy(center).float())
            tree_prototypes[key] = prototypes
        else:
            # Just use mean as single prototype
            tree_prototypes[key] = [tree_embeddings[key]]

        tree_stats[key] = {
            'total': len(tree_embs),
            'kept': int(keep_mask.sum()),
            'removed': int(n_removed),
            'prototypes': len(tree_prototypes[key])
        }

    print(f"\nOutliers removed: {outliers_removed}/{total_photos} ({outliers_removed/total_photos*100:.1f}%)")
    print(f"Trees processed: {len(tree_embeddings)}")

    # Stats on prototypes
    proto_counts = [len(p) for p in tree_prototypes.values()]
    print(f"Prototypes per tree: min={min(proto_counts)}, max={max(proto_counts)}, avg={np.mean(proto_counts):.1f}")

    # Save
    torch.save({
        'embeddings': tree_embeddings,          # Single mean per tree
        'prototypes': tree_prototypes,          # Multiple prototypes per tree
        'stats': tree_stats,
        'feature_dim': feature_dim,
        'encoder': encoder,
        'config': {
            'n_prototypes': n_prototypes,
            'outlier_threshold': outlier_threshold
        }
    }, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\model_seed42.pt')
    parser.add_argument('--data', default=r'E:\tree_id_2.0\data')
    parser.add_argument('--images', default=r'E:\tree_id_2.0\images')
    parser.add_argument('--output', default=r'E:\tree_id_2.0\models\tree_embeddings_v2.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prototypes', type=int, default=3, help='Max prototypes per tree')
    parser.add_argument('--outlier_threshold', type=float, default=0.5,
                       help='Min similarity to mean to keep (0-1)')
    parser.add_argument('--input', default='training_data_with_ground_truth.xlsx',
                       help='Input data file name')
    args = parser.parse_args()

    compute_embeddings_v2(args.model, args.data, args.images, args.output,
                         args.batch_size, args.prototypes, args.outlier_threshold,
                         args.input)
