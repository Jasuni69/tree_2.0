"""
Train ConvNeXt-Base at 384x384 resolution for tree re-identification.
Higher resolution captures fine bark/branch details for same-address discrimination.

Changes from train_metric.py:
- Input: 384x384 (was 224x224)
- Batch: P=8 K=4 = 32 (was P=16 K=4 = 64) - fits in VRAM
- Output: models/metric_384/
- Loads from previous best 224 model as init

Run: python training/train_metric_384.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math

from dataset_metric import TreeMetricDataset, AddressAwarePKSampler
from model_metric import TreeReIdModel
from losses_metric import SubCenterArcFace, BatchHardTripletLoss
from progress_window import TrainingProgressWindow


def compute_recall_at_k(embeddings, labels, k_values=[1, 5]):
    embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1)
    sim_matrix = torch.mm(embeddings, embeddings.t())
    sim_matrix.fill_diagonal_(-1)

    results = {}
    for k in k_values:
        correct = 0
        total = 0
        _, topk_indices = sim_matrix.topk(k, dim=1)
        for i in range(len(labels)):
            topk_labels = [labels[j] for j in topk_indices[i].tolist()]
            if labels[i] in topk_labels:
                correct += 1
            total += 1
        results[f'recall@{k}'] = correct / total if total > 0 else 0
    return results


def validate(model, val_dataset, device, batch_size=32):
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val', leave=False):
            imgs = imgs.to(device)
            emb = model(imgs)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels.tolist())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    results = compute_recall_at_k(all_embeddings, all_labels, k_values=[1, 5])
    return results


def train():
    # Config
    INIT_CKPT = r'E:\tree_id_2.0\models\metric\best_model.pth'  # init from 224 model
    DATA_DIR = r'E:\tree_id_2.0\data'
    IMAGE_BASE = r'E:\tree_id_2.0\images'
    OUTPUT_DIR = r'E:\tree_id_2.0\models\metric_384'
    EXCEL = 'training_data_cleaned.xlsx'
    ENCODER = 'label_encoder_gt.json'

    # Model
    EMBEDDING_DIM = 1024
    INPUT_SIZE = 384

    # Training - smaller batch for 384 VRAM
    EPOCHS = 60
    BATCH_P = 8
    BATCH_K = 4
    LR_BACKBONE = 5e-6   # lower LR - fine-tuning from good init
    LR_HEAD = 2e-4
    LR_ARCFACE = 5e-4
    WEIGHT_DECAY = 1e-4
    FREEZE_EPOCHS = 3
    WARMUP_EPOCHS = 2

    # Loss
    ARCFACE_MARGIN = 0.5
    ARCFACE_SCALE = 30.0
    NUM_SUBCENTERS = 3
    TRIPLET_MARGIN = 0.5
    TRIPLET_WEIGHT = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Resolution: {INPUT_SIZE}x{INPUT_SIZE}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Data - with 384 input size
    excel_path = str(Path(DATA_DIR) / EXCEL)
    encoder_path = str(Path(DATA_DIR) / ENCODER)

    train_dataset = TreeMetricDataset(excel_path, IMAGE_BASE, encoder_path,
                                      split='train', input_size=INPUT_SIZE)
    val_dataset = TreeMetricDataset(excel_path, IMAGE_BASE, encoder_path,
                                    split='val', input_size=INPUT_SIZE)

    sampler = AddressAwarePKSampler(train_dataset, p=BATCH_P, k=BATCH_K)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=3)

    # Log address stats
    print(f"\nAddress-aware sampling stats:")
    addr_sizes = [len(v) for v in sampler.address_to_trees.values()]
    print(f"  Total addresses: {len(sampler.address_to_trees)}")
    print(f"  Multi-tree addresses: {len(sampler.multi_tree_addresses)}")
    print(f"  Trees per address: min={min(addr_sizes)}, max={max(addr_sizes)}, "
          f"mean={np.mean(addr_sizes):.1f}")

    # Model - init from 224 trained model
    print(f"\nLoading init from {INIT_CKPT}")
    model = TreeReIdModel.from_pretrained_partial(INIT_CKPT, device=str(device))
    model.train()

    # Count classes
    with open(encoder_path) as f:
        encoder = json.load(f)
    num_classes = len(encoder)
    print(f"Num classes: {num_classes}")

    # Losses
    arcface_loss = SubCenterArcFace(
        in_features=EMBEDDING_DIM,
        num_classes=num_classes,
        num_subcenters=NUM_SUBCENTERS,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
    ).to(device)

    # Load ArcFace weights from previous training if available
    init_ckpt = torch.load(INIT_CKPT, map_location=device)
    if 'arcface_state_dict' in init_ckpt:
        arcface_loss.load_state_dict(init_ckpt['arcface_state_dict'])
        print("Loaded ArcFace weights from init checkpoint")

    triplet_loss = BatchHardTripletLoss(margin=TRIPLET_MARGIN)

    # Optimizer - differential LR
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.embedding_head.parameters()) + list(model.pool.parameters())
    arcface_params = list(arcface_loss.parameters())

    # Phase 1: freeze backbone
    for p in backbone_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': [p for p in backbone_params if p.requires_grad], 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD},
        {'params': arcface_params, 'lr': LR_ARCFACE},
    ], weight_decay=WEIGHT_DECAY)

    scaler = GradScaler('cuda')
    best_recall = 0.0

    # Live progress window
    progress = TrainingProgressWindow(total_epochs=EPOCHS)

    print(f"\n{'='*60}")
    print(f"Training 384x384: {EPOCHS} epochs, P={BATCH_P} K={BATCH_K}")
    print(f"Phase 1 (frozen backbone): epochs 1-{FREEZE_EPOCHS}")
    print(f"Phase 2 (full fine-tune): epochs {FREEZE_EPOCHS+1}-{EPOCHS}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        # Phase transition: unfreeze backbone
        if epoch == FREEZE_EPOCHS + 1:
            print("\n>>> Unfreezing backbone for full fine-tuning")
            progress.update_status("Phase 2 (full fine-tune)")
            model.unfreeze_all()
            optimizer = torch.optim.AdamW([
                {'params': list(model.backbone.parameters()), 'lr': LR_BACKBONE},
                {'params': head_params, 'lr': LR_HEAD},
                {'params': arcface_params, 'lr': LR_ARCFACE},
            ], weight_decay=WEIGHT_DECAY)
            scaler = GradScaler('cuda')

        # Cosine LR schedule (after warmup)
        if epoch <= WARMUP_EPOCHS:
            lr_scale = epoch / WARMUP_EPOCHS
        else:
            cos_progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * cos_progress))
            lr_scale = max(lr_scale, 0.01)

        for pg in optimizer.param_groups:
            if pg['lr'] == LR_BACKBONE or (epoch > FREEZE_EPOCHS and pg is optimizer.param_groups[0]):
                pg['lr'] = LR_BACKBONE * lr_scale
            elif pg is optimizer.param_groups[1]:
                pg['lr'] = LR_HEAD * lr_scale
            else:
                pg['lr'] = LR_ARCFACE * lr_scale

        # Train
        model.train()
        sampler.set_epoch(epoch)

        epoch_arc_loss = 0
        epoch_tri_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                embeddings = model(imgs)
                loss_arc = arcface_loss(embeddings, labels)
                loss_tri = triplet_loss(embeddings, labels)
                loss = loss_arc + TRIPLET_WEIGHT * loss_tri

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_arc_loss += loss_arc.item()
            epoch_tri_loss += loss_tri.item()
            num_batches += 1

            pbar.set_postfix({
                'arc': f'{loss_arc.item():.3f}',
                'tri': f'{loss_tri.item():.3f}',
                'lr': f'{optimizer.param_groups[1]["lr"]:.1e}'
            })

            if num_batches % 5 == 0:
                progress.update_batch(num_batches, len(train_loader),
                                      arc_loss=loss_arc.item(),
                                      tri_loss=loss_tri.item())

        avg_arc = epoch_arc_loss / max(num_batches, 1)
        avg_tri = epoch_tri_loss / max(num_batches, 1)
        print(f"Epoch {epoch}: arc_loss={avg_arc:.4f}, tri_loss={avg_tri:.4f}")
        progress.update_epoch(epoch, arc_loss=avg_arc, tri_loss=avg_tri,
                              lr=optimizer.param_groups[1]['lr'])

        # Validate every 2 epochs (or last epoch)
        if epoch % 2 == 0 or epoch == EPOCHS:
            recall = validate(model, val_dataset, device, batch_size=32)
            r1 = recall['recall@1']
            r5 = recall['recall@5']
            print(f"  Val: recall@1={r1:.4f}, recall@5={r5:.4f}")
            progress.update_val(recall_1=r1, recall_5=r5)

            if r1 > best_recall:
                best_recall = r1
                save_path = Path(OUTPUT_DIR) / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'arcface_state_dict': arcface_loss.state_dict(),
                    'best_recall': best_recall,
                    'config': {
                        'backbone_name': 'convnext_base',
                        'embedding_dim': EMBEDDING_DIM,
                        'input_size': INPUT_SIZE,
                        'dropout_rate': 0.3,
                        'pooling': 'gem',
                    }
                }, save_path)
                print(f"  Saved best model (recall@1={r1:.4f})")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = Path(OUTPUT_DIR) / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'arcface_state_dict': arcface_loss.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'backbone_name': 'convnext_base',
                    'embedding_dim': EMBEDDING_DIM,
                    'input_size': INPUT_SIZE,
                    'dropout_rate': 0.3,
                    'pooling': 'gem',
                }
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\nTraining complete. Best recall@1: {best_recall:.4f}")
    print(f"Best model: {Path(OUTPUT_DIR) / 'best_model.pth'}")
    progress.update_status(f"Done! Best recall@1: {best_recall:.2%}")
    import time; time.sleep(5)
    progress.close()


if __name__ == '__main__':
    train()
