"""
Tree identification model training.

Uses EfficientNet backbone with classification head.
Windows-safe DataLoader settings included.

Run single: python train.py --seed 42
Run ensemble: python train.py --ensemble
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import timm
from tqdm import tqdm
import json
import gc
import argparse
import numpy as np

from dataset import TreeDataset, prepare_data, get_transforms


class TreeClassifier(nn.Module):
    """EfficientNet-based tree classifier."""

    def __init__(self, num_classes, backbone='efficientnet_b0', pretrained=True):
        super().__init__()

        # Load pretrained backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        """Extract feature embeddings without classification."""
        return self.backbone(x)


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def train_single(config, seed):
    """Train a single model with given seed."""
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Training with seed {seed}")
    print(f"Using device: {device}")
    print(f"{'='*50}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Prepare data (use Tibram ground truth)
    print("\n=== Loading Data ===")
    print(f"Using input file: {config['input_file']}")
    df, encoder, decoder = prepare_data(
        config['data_path'],
        outlier_threshold_m=config['outlier_threshold_m'],
        use_ground_truth=True,
        input_file=config['input_file']
    )

    num_classes = len(encoder)
    print(f"Number of classes: {num_classes}")

    # Save encoder/decoder (only once, not per seed)
    encoder_file = output_dir / 'label_encoder.json'
    if not encoder_file.exists():
        with open(encoder_file, 'w') as f:
            if isinstance(list(encoder.keys())[0], tuple):
                encoder_json = {f"{k[0]}|||{k[1]}": v for k, v in encoder.items()}
            else:
                encoder_json = encoder
            json.dump(encoder_json, f)

        with open(output_dir / 'label_decoder.json', 'w') as f:
            if isinstance(list(decoder.values())[0], tuple):
                decoder_json = {str(k): f"{v[0]}|||{v[1]}" for k, v in decoder.items()}
            else:
                decoder_json = {str(k): v for k, v in decoder.items()}
            json.dump(decoder_json, f)

    # Create transforms
    train_transform = get_transforms(train=True, img_size=config['img_size'])
    val_transform = get_transforms(train=False, img_size=config['img_size'])

    # Create SEPARATE datasets for train and val (fixes augmentation bug)
    train_full = TreeDataset(df, config['image_base'], transform=train_transform, label_encoder=encoder)
    val_full = TreeDataset(df, config['image_base'], transform=val_transform, label_encoder=encoder)

    # Generate deterministic split indices
    n_samples = len(train_full)
    indices = list(range(n_samples))

    # Shuffle with seed
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    val_size = int(n_samples * config['val_split'])
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Create subsets with correct transforms
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

    # DataLoaders - Windows safe settings with larger batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )

    # Model
    print("\n=== Creating Model ===")
    model = TreeClassifier(num_classes, backbone=config['backbone'], pretrained=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Training loop
    print("\n=== Training ===")
    best_val_acc = 0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        # Step scheduler
        scheduler.step()

        # Save best model (with seed in filename)
        model_name = f'model_seed{seed}.pt'
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'seed': seed,
                'config': config
            }, output_dir / model_name)
            print(f"Saved best model (val_acc: {val_acc*100:.2f}%)")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'seed': seed,
            'config': config
        }, output_dir / f'checkpoint_seed{seed}.pt')

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n=== Training Complete (seed {seed}) ===")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {output_dir / model_name}")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train tree classifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ensemble', action='store_true', help='Train 5 models with different seeds')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--backbone', type=str, default='efficientnet_b2', help='Model backbone')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=260, help='Image size (default: 260 for b2)')
    parser.add_argument('--input', type=str, default='training_data_with_ground_truth.xlsx', help='Input data file')
    args = parser.parse_args()

    # Config
    config = {
        'data_path': r'E:\tree_id_2.0\data',
        'image_base': r'E:\tree_id_2.0\images',
        'output_dir': r'E:\tree_id_2.0\models',
        'backbone': args.backbone,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'outlier_threshold_m': 15,
        'val_split': 0.15,
        'input_file': args.input,
    }

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    if args.ensemble:
        # Train 5 models with different seeds
        seeds = [42, 123, 456, 789, 1337]
        results = {}

        print(f"\n{'='*50}")
        print(f"ENSEMBLE TRAINING: {len(seeds)} models")
        print(f"Seeds: {seeds}")
        print(f"{'='*50}")

        for seed in seeds:
            val_acc = train_single(config, seed)
            results[seed] = val_acc

        print(f"\n{'='*50}")
        print("ENSEMBLE RESULTS")
        print(f"{'='*50}")
        for seed, acc in results.items():
            print(f"  Seed {seed}: {acc*100:.2f}%")
        print(f"  Mean: {np.mean(list(results.values()))*100:.2f}%")
        print(f"  Std:  {np.std(list(results.values()))*100:.2f}%")
    else:
        # Train single model
        train_single(config, args.seed)


if __name__ == '__main__':
    main()
