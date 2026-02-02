"""
Dataset and PK batch sampler for metric learning training.
Loads from training_data_cleaned.xlsx with image paths relative to image_base.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Iterator
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image


class TreeMetricDataset(Dataset):
    """Dataset for metric learning. Returns (image, tree_id)."""

    def __init__(self, excel_path: str, image_base: str, encoder_path: str,
                 split: str = 'train', train_ratio: float = 0.8,
                 val_ratio: float = 0.1, seed: int = 42,
                 input_size: int = 224):
        self.image_base = Path(image_base)
        self.input_size = input_size

        # Load data
        df = pd.read_excel(excel_path)
        with open(encoder_path) as f:
            self.encoder = json.load(f)

        # Tree-level split
        all_keys = sorted(df['key'].unique())
        rng = np.random.RandomState(seed)
        shuffled = all_keys.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == 'train':
            keep_keys = set(shuffled[:train_end])
        elif split == 'val':
            keep_keys = set(shuffled[train_end:val_end])
        else:
            keep_keys = set(shuffled[val_end:])

        self.df = df[df['key'].isin(keep_keys)].reset_index(drop=True)
        self.split = split

        # Build tree_id -> dataset indices mapping
        self._tree_to_indices = defaultdict(list)
        self._unique_tree_ids = set()

        for idx, row in self.df.iterrows():
            key = row['key']
            if key in self.encoder:
                tree_id = self.encoder[key]
                self._tree_to_indices[tree_id].append(idx)
                self._unique_tree_ids.add(tree_id)

        # Transforms
        resize_size = int(input_size * 256 / 224)  # keep same resize/crop ratio
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        print(f"[{split}] {len(self.df)} photos, {len(self._unique_tree_ids)} trees")

    @property
    def unique_tree_ids(self):
        return self._unique_tree_ids

    def get_tree_indices(self, tree_id: int) -> List[int]:
        return self._tree_to_indices[tree_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_base / row['image_path']
        tree_id = self.encoder[row['key']]

        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, self.input_size, self.input_size)

        return img, tree_id


class PKBatchSampler(Sampler):
    """
    P×K batch sampler: P trees, K photos per tree.
    Trees with <K photos are sampled with replacement.
    """
    def __init__(self, dataset: TreeMetricDataset,
                 p: int = 8, k: int = 4, seed: int = 42):
        self.dataset = dataset
        self.P = p
        self.K = k
        self.seed = seed
        self.epoch = 0

        self.tree_ids = sorted(list(dataset.unique_tree_ids))
        self.tree_to_indices = {}
        for tid in self.tree_ids:
            self.tree_to_indices[tid] = dataset.get_tree_indices(tid)

        # Filter trees with at least 1 photo
        self.tree_ids = [t for t in self.tree_ids if len(self.tree_to_indices[t]) > 0]
        self.num_batches = len(dataset) // (self.P * self.K)
        print(f"PKSampler: P={p}, K={k}, batch={p*k}, batches/epoch={self.num_batches}, trees={len(self.tree_ids)}")

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)

        for _ in range(self.num_batches):
            batch = []
            sampled_trees = rng.choice(self.tree_ids, size=min(self.P, len(self.tree_ids)), replace=False)

            for tid in sampled_trees:
                indices = self.tree_to_indices[tid]
                if len(indices) >= self.K:
                    chosen = rng.choice(indices, size=self.K, replace=False)
                else:
                    chosen = rng.choice(indices, size=self.K, replace=True)
                batch.extend(chosen.tolist())

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class AddressAwarePKSampler(Sampler):
    """
    Address-aware P×K batch sampler.
    Picks trees from same address so triplet loss mines hard negatives
    (same-address trees that look similar).
    """
    def __init__(self, dataset: TreeMetricDataset,
                 p: int = 16, k: int = 4, seed: int = 42):
        self.dataset = dataset
        self.P = p
        self.K = k
        self.seed = seed
        self.epoch = 0

        # Build tree_id -> indices mapping
        self.tree_ids = sorted(list(dataset.unique_tree_ids))
        self.tree_to_indices = {}
        for tid in self.tree_ids:
            self.tree_to_indices[tid] = dataset.get_tree_indices(tid)
        self.tree_ids = [t for t in self.tree_ids if len(self.tree_to_indices[t]) > 0]

        # Build key -> tree_id mapping from encoder
        self.key_to_tree_id = {}
        for key, tid in dataset.encoder.items():
            if tid in dataset.unique_tree_ids:
                self.key_to_tree_id[key] = tid

        # Build address -> tree_ids mapping
        self.address_to_trees = defaultdict(set)
        for key, tid in self.key_to_tree_id.items():
            addr = key.rsplit('|', 1)[0]
            if tid in self.tree_to_indices and len(self.tree_to_indices[tid]) > 0:
                self.address_to_trees[addr].add(tid)

        # Convert sets to sorted lists
        self.address_to_trees = {
            addr: sorted(list(tids))
            for addr, tids in self.address_to_trees.items()
        }

        # Filter to addresses with >=2 trees (need negatives)
        self.multi_tree_addresses = [
            addr for addr, tids in self.address_to_trees.items()
            if len(tids) >= 2
        ]
        self.multi_tree_addresses.sort()

        # All tree_ids at multi-tree addresses (for filling)
        self.fill_pool = []
        for addr in self.multi_tree_addresses:
            self.fill_pool.extend(self.address_to_trees[addr])
        self.fill_pool = sorted(list(set(self.fill_pool)))

        self.num_batches = len(dataset) // (self.P * self.K)

        print(f"AddressAwarePKSampler: P={p}, K={k}, batch={p*k}, "
              f"batches/epoch={self.num_batches}")
        print(f"  Addresses with >=2 trees: {len(self.multi_tree_addresses)}, "
              f"trees in pool: {len(self.fill_pool)}")

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)

        for _ in range(self.num_batches):
            batch = []

            # Pick random address with >=2 trees
            addr = self.multi_tree_addresses[
                rng.randint(len(self.multi_tree_addresses))
            ]
            addr_trees = self.address_to_trees[addr]

            if len(addr_trees) >= self.P:
                # Address has enough trees, sample P from it
                sampled = rng.choice(addr_trees, size=self.P, replace=False)
            else:
                # Take all trees at this address, fill rest from other addresses
                sampled = list(addr_trees)
                remaining = self.P - len(sampled)
                # Pick fill trees not already in batch
                available = [t for t in self.fill_pool if t not in sampled]
                if len(available) >= remaining:
                    fill = rng.choice(available, size=remaining, replace=False)
                else:
                    fill = rng.choice(available, size=remaining, replace=True)
                sampled.extend(fill.tolist())

            # Sample K photos per tree
            for tid in sampled:
                indices = self.tree_to_indices[tid]
                if len(indices) >= self.K:
                    chosen = rng.choice(indices, size=self.K, replace=False)
                else:
                    chosen = rng.choice(indices, size=self.K, replace=True)
                batch.extend(chosen.tolist())

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch
