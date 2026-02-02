"""
Precompute average embeddings for each tree.

Run once after training to generate tree_embeddings.pt
Uses batched processing for speed.
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


class TreeClassifier(nn.Module):
    def __init__(self, num_classes, backbone='efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x):
        return self.backbone(x)


class ImageDataset(Dataset):
    """Simple dataset for batch processing."""

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
            # Return dummy image on error
            return torch.zeros(3, 260, 260), idx, False


def compute_embeddings(model_path, data_dir, image_base, output_path, batch_size=64):
    """Compute average embedding for each tree using batched processing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load data
    with open(data_dir / 'label_encoder_gt.json') as f:
        encoder = json.load(f)
    num_classes = len(encoder)

    df = pd.read_excel(data_dir / 'training_data_with_ground_truth.xlsx')
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

    # Process ALL images in batches, then aggregate by tree
    all_paths = df['image_path'].tolist()
    all_keys = df['key'].tolist()

    dataset = ImageDataset(all_paths, image_base, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                       pin_memory=True, shuffle=False)

    # Extract all embeddings
    all_embeddings = []
    valid_indices = []

    print(f"\nExtracting embeddings (batch_size={batch_size})...")
    with torch.no_grad():
        for batch_imgs, batch_idx, batch_valid in tqdm(loader, desc='Extracting'):
            batch_imgs = batch_imgs.to(device)
            features = model.extract_features(batch_imgs)
            features = F.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu())
            valid_indices.extend([(i.item(), v.item()) for i, v in zip(batch_idx, batch_valid)])

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Aggregate by tree
    print("\nAggregating by tree...")
    tree_embeddings = {}
    tree_counts = {}

    for key in tqdm(df['key'].unique(), desc='Aggregating'):
        # Get indices for this tree
        tree_mask = df['key'] == key
        indices = df[tree_mask].index.tolist()

        # Filter valid embeddings
        valid_embs = []
        for idx in indices:
            # Find position in our processed list
            if idx < len(valid_indices) and valid_indices[idx][1]:
                valid_embs.append(all_embeddings[idx])

        if valid_embs:
            # Average and normalize
            avg_emb = torch.stack(valid_embs).mean(dim=0)
            avg_emb = F.normalize(avg_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
            tree_embeddings[key] = avg_emb
            tree_counts[key] = len(valid_embs)

    print(f"\nComputed embeddings for {len(tree_embeddings)} trees")
    print(f"Avg photos per tree: {np.mean(list(tree_counts.values())):.1f}")

    # Save
    torch.save({
        'embeddings': tree_embeddings,
        'counts': tree_counts,
        'feature_dim': feature_dim,
        'encoder': encoder
    }, output_path)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\model_seed42.pt')
    parser.add_argument('--data', default=r'E:\tree_id_2.0\data')
    parser.add_argument('--images', default=r'E:\tree_id_2.0\images')
    parser.add_argument('--output', default=r'E:\tree_id_2.0\models\tree_embeddings.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    compute_embeddings(args.model, args.data, args.images, args.output, args.batch_size)
