"""
Tree identification dataset for training.

Loads images and maps them to tree IDs using pre-extracted metadata.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np


class TreeDataset(Dataset):
    """Dataset for tree identification training."""

    def __init__(self, df, image_base, transform=None, label_encoder=None):
        """
        Args:
            df: DataFrame with image_path and tree identification columns
            image_base: Base path for images
            transform: torchvision transforms
            label_encoder: Dict mapping key -> class_id (optional if df has tree_id)
        """
        self.df = df.reset_index(drop=True)
        self.image_base = Path(image_base)
        self.transform = transform
        self.label_encoder = label_encoder

        # Create tree_id column if not present
        if 'tree_id' not in self.df.columns and label_encoder:
            if 'key' in self.df.columns:
                # Ground truth format
                self.df['tree_id'] = self.df['key'].map(label_encoder).fillna(-1).astype(int)
            else:
                # Legacy format
                self.df['tree_id'] = self.df.apply(
                    lambda row: label_encoder.get((row['address'], row['tree_number']), -1),
                    axis=1
                )
            # Filter out unknown trees
            self.df = self.df[self.df['tree_id'] >= 0].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.image_base / row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return black image on error
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Get label
        label = row['tree_id'] if 'tree_id' in row else 0

        # Get coordinates - prefer ground truth if available
        if 'gt_lat' in row and pd.notna(row['gt_lat']):
            lat = row['gt_lat']
            lon = row['gt_lon']
        else:
            lat = row['photo_lat'] if 'photo_lat' in row else 0
            lon = row['photo_lon'] if 'photo_lon' in row else 0

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'coords': torch.tensor([lat, lon], dtype=torch.float32),
            'idx': idx
        }


def create_label_encoder(df):
    """Create mapping from (address, tree_number) to class_id."""
    unique_trees = df[['address', 'tree_number']].drop_duplicates()
    encoder = {}
    decoder = {}

    for i, (_, row) in enumerate(unique_trees.iterrows()):
        key = (row['address'], row['tree_number'])
        encoder[key] = i
        decoder[i] = key

    return encoder, decoder


def prepare_data(data_path, outlier_threshold_m=15, use_ground_truth=True, input_file='training_data_with_ground_truth.xlsx'):
    """
    Load and prepare data for training.

    Args:
        data_path: Path to data directory
        outlier_threshold_m: Remove photos > this distance from ground truth/median
        use_ground_truth: If True, use Tibram ground truth data
        input_file: Name of input Excel file (default: training_data_with_ground_truth.xlsx)

    Returns:
        df: Cleaned DataFrame
        encoder: Label encoder dict
        decoder: Label decoder dict
    """
    data_path = Path(data_path)

    if use_ground_truth:
        # Use pre-prepared ground truth data
        gt_file = data_path / input_file
        if gt_file.exists():
            df = pd.read_excel(gt_file)
            print(f"Loaded ground truth data: {len(df)} photos")
            print(f"Unique trees: {df['key'].nunique()}")

            # Load encoder
            import json
            encoder_file = data_path / 'label_encoder_gt.json'
            if encoder_file.exists():
                with open(encoder_file) as f:
                    encoder = json.load(f)
                decoder = {v: k for k, v in encoder.items()}
            else:
                encoder = {key: i for i, key in enumerate(sorted(df['key'].unique()))}
                decoder = {i: key for key, i in encoder.items()}

            # Add tree_id for dataset
            df['tree_id'] = df['key'].map(encoder)

            print(f"Classes: {len(encoder)}")
            return df, encoder, decoder
        else:
            print("Ground truth file not found, falling back to median-based data")

    # Fallback: Load outliers data (median-based)
    df = pd.read_excel(data_path / 'photos_by_tree_with_outliers.xlsx')

    print(f"Total photos: {len(df)}")
    print(f"Unique trees: {df.groupby(['address', 'tree_number']).ngroups}")

    # Filter outliers
    if outlier_threshold_m:
        before = len(df)
        df = df[df['distance_from_tree_median_m'] <= outlier_threshold_m]
        print(f"After filtering >{outlier_threshold_m}m outliers: {len(df)} ({len(df)/before*100:.1f}%)")

    # Create label encoder
    encoder, decoder = create_label_encoder(df)
    print(f"Classes (unique trees): {len(encoder)}")

    return df, encoder, decoder


def get_transforms(train=True, img_size=224):
    """Get image transforms for training/validation."""
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    # Test dataset
    DATA_PATH = Path(r'E:\tree_id_2.0\data')
    IMAGE_BASE = Path(r'E:\tree_id_2.0\images')

    df, encoder, decoder = prepare_data(DATA_PATH, outlier_threshold_m=15)

    # Create dataset
    transform = get_transforms(train=True)
    dataset = TreeDataset(df, IMAGE_BASE, transform=transform, label_encoder=encoder)

    print(f"\nDataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")
    print(f"Coords: {sample['coords']}")
