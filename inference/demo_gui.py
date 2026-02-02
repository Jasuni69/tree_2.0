"""
Real-time inference demo GUI for tree re-identification.
Uses 15% holdout set for fair demonstrations.

Run: python inference/demo_gui.py
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
import math
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from model_metric import TreeReIdModel


class TreeDemoGUI:
    def __init__(self, model_path: str, data_dir: str, image_base: str,
                 holdout_ratio: float = 0.15, radius_m: float = 30):
        self.data_dir = Path(data_dir)
        self.image_base = Path(image_base)
        self.radius_m = radius_m
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        self.model = TreeReIdModel(
            backbone_name=config.get('backbone_name', 'convnext_base'),
            embedding_dim=config.get('embedding_dim', 1024),
            pretrained=False, freeze_stages=0,
            dropout_rate=config.get('dropout_rate', 0.3),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load data and create holdout split
        print("Loading data and building prototypes...")
        self._load_data(holdout_ratio)
        self._build_prototypes()
        print(f"Ready! {len(self.holdout_indices)} holdout photos, {len(self.tree_prototypes)} trees with prototypes")

        # Create GUI
        self._create_gui()

    def _load_data(self, holdout_ratio: float):
        self.df = pd.read_excel(self.data_dir / 'training_data_cleaned.xlsx')

        # Stratified split - same as evaluate_metric.py
        np.random.seed(42)
        train_indices, holdout_indices = [], []
        for key, group in self.df.groupby('key'):
            indices = group.index.tolist()
            if len(indices) < 2:
                train_indices.extend(indices)
                continue
            np.random.shuffle(indices)
            n_holdout = max(1, int(len(indices) * holdout_ratio))
            holdout_indices.extend(indices[:n_holdout])
            train_indices.extend(indices[n_holdout:])

        self.train_indices = set(train_indices)
        self.holdout_indices = holdout_indices

        # GPS coords
        trees = self.df.groupby('key').agg({'gt_lat': 'first', 'gt_lon': 'first'}).reset_index()
        self.tree_coords = {r['key']: (r['gt_lat'], r['gt_lon']) for _, r in trees.iterrows()}

        # Address mapping
        self.address_trees = defaultdict(list)
        for key in self.df['key'].unique():
            addr = key.rsplit('|', 1)[0] if '|' in key else key
            self.address_trees[addr].append(key)

    def _build_prototypes(self, n_prototypes: int = 3, outlier_threshold: float = 0.5):
        # Extract embeddings for train set
        all_embeddings = {}
        with torch.no_grad():
            for idx in self.train_indices:
                row = self.df.iloc[idx]
                img_path = self.image_base / row['image_path']
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = self.transform(img).unsqueeze(0).to(self.device)
                    emb = self.model(img_t).cpu().numpy()[0]
                    all_embeddings[idx] = emb
                except:
                    continue

        # Build prototypes per tree
        self.tree_prototypes = {}
        for key, group in self.df.groupby('key'):
            indices = [i for i in group.index.tolist() if i in all_embeddings]
            if not indices:
                continue

            embs = np.array([all_embeddings[i] for i in indices])
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

            self.tree_prototypes[key] = protos

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _create_gui(self):
        self.root = tk.Tk()
        self.root.title("Tree Re-ID Demo")
        self.root.geometry("900x1200")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"),
                        foreground="#eee", background="#1a1a2e")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12),
                        foreground="#888", background="#1a1a2e")
        style.configure("Result.TLabel", font=("Consolas", 14),
                        foreground="#4ecca3", background="#1a1a2e")
        style.configure("Info.TLabel", font=("Consolas", 11),
                        foreground="#aaa", background="#1a1a2e")
        style.configure("Correct.TLabel", font=("Segoe UI", 16, "bold"),
                        foreground="#4ecca3", background="#1a1a2e")
        style.configure("Wrong.TLabel", font=("Segoe UI", 16, "bold"),
                        foreground="#ff6b6b", background="#1a1a2e")
        style.configure("Big.TButton", font=("Segoe UI", 14), padding=15)

        main = tk.Frame(self.root, bg="#1a1a2e", padx=30, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main, text="Tree Re-Identification", style="Title.TLabel").pack(pady=(0, 5))
        ttk.Label(main, text="Testing on 15% holdout set - images model has never seen",
                  style="Subtitle.TLabel").pack(pady=(0, 20))

        # Image display area
        img_frame = tk.Frame(main, bg="#252540", width=500, height=500)
        img_frame.pack(pady=10)
        img_frame.pack_propagate(False)
        self.img_label = tk.Label(img_frame, bg="#252540")
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")

        # Ground truth
        gt_frame = tk.Frame(main, bg="#1a1a2e")
        gt_frame.pack(pady=15, fill=tk.X)
        ttk.Label(gt_frame, text="Ground Truth:", style="Info.TLabel").pack(side=tk.LEFT)
        self.gt_label = ttk.Label(gt_frame, text="--", style="Result.TLabel")
        self.gt_label.pack(side=tk.LEFT, padx=10)

        # Prediction
        pred_frame = tk.Frame(main, bg="#1a1a2e")
        pred_frame.pack(pady=5, fill=tk.X)
        ttk.Label(pred_frame, text="Prediction: ", style="Info.TLabel").pack(side=tk.LEFT)
        self.pred_label = ttk.Label(pred_frame, text="--", style="Result.TLabel")
        self.pred_label.pack(side=tk.LEFT, padx=10)

        # Similarity score
        sim_frame = tk.Frame(main, bg="#1a1a2e")
        sim_frame.pack(pady=5, fill=tk.X)
        ttk.Label(sim_frame, text="Similarity:  ", style="Info.TLabel").pack(side=tk.LEFT)
        self.sim_label = ttk.Label(sim_frame, text="--", style="Result.TLabel")
        self.sim_label.pack(side=tk.LEFT, padx=10)

        # Candidates
        cand_frame = tk.Frame(main, bg="#1a1a2e")
        cand_frame.pack(pady=5, fill=tk.X)
        ttk.Label(cand_frame, text="Candidates: ", style="Info.TLabel").pack(side=tk.LEFT)
        self.cand_label = ttk.Label(cand_frame, text="--", style="Info.TLabel")
        self.cand_label.pack(side=tk.LEFT, padx=10)

        # Result (correct/wrong)
        self.result_label = ttk.Label(main, text="", style="Correct.TLabel")
        self.result_label.pack(pady=20)

        # Top 3 matches
        top3_title = ttk.Label(main, text="Top 3 Matches", style="Subtitle.TLabel")
        top3_title.pack(pady=(10, 5))

        self.top3_frame = tk.Frame(main, bg="#1a1a2e")
        self.top3_frame.pack(pady=5, fill=tk.X)
        self.top3_labels = []
        for i in range(3):
            lbl = ttk.Label(self.top3_frame, text=f"{i+1}. --", style="Info.TLabel")
            lbl.pack(anchor="w", pady=2)
            self.top3_labels.append(lbl)

        # Image selector
        selector_frame = tk.Frame(main, bg="#1a1a2e")
        selector_frame.pack(pady=10, fill=tk.X)

        ttk.Label(selector_frame, text=f"Select from {len(self.holdout_indices)} holdout images (15% - NEVER seen during training):",
                  style="Info.TLabel").pack(anchor="w")

        # Listbox with scrollbar
        list_frame = tk.Frame(selector_frame, bg="#1a1a2e")
        list_frame.pack(fill=tk.X, pady=5)

        self.image_listbox = tk.Listbox(list_frame, height=6, width=70,
                                         bg="#252540", fg="#ccc", font=("Consolas", 10),
                                         selectbackground="#4ecca3", selectforeground="#000")
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.image_listbox.yview)

        self.image_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate listbox with holdout images
        self.holdout_data = []
        for idx in self.holdout_indices:
            row = self.df.iloc[idx]
            display = f"{row['key']} | {row['image_path']}"
            self.image_listbox.insert(tk.END, display)
            self.holdout_data.append(idx)

        self.image_listbox.bind('<<ListboxSelect>>', self._on_select)

        # Buttons
        btn_frame = tk.Frame(main, bg="#1a1a2e")
        btn_frame.pack(pady=15)

        self.next_btn = ttk.Button(btn_frame, text="Random Image",
                                   style="Big.TButton", command=self._next_image)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        self.auto_btn = ttk.Button(btn_frame, text="Auto Demo (3s)",
                                   style="Big.TButton", command=self._toggle_auto)
        self.auto_btn.pack(side=tk.LEFT, padx=10)

        # Stats
        self.stats_label = ttk.Label(main, text="Tested: 0 | Correct: 0 | Accuracy: --",
                                     style="Subtitle.TLabel")
        self.stats_label.pack(pady=10)

        self.tested = 0
        self.correct = 0
        self.auto_running = False

        # Show first image
        self._next_image()

    def _on_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            list_idx = selection[0]
            idx = self.holdout_data[list_idx]
            self._show_image(idx)

    def _next_image(self):
        # Pick random holdout image
        idx = random.choice(self.holdout_indices)
        # Highlight in listbox
        list_idx = self.holdout_data.index(idx)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(list_idx)
        self.image_listbox.see(list_idx)
        self._show_image(idx)

    def _show_image(self, idx):
        row = self.df.iloc[idx]
        true_key = row['key']
        img_path = self.image_base / row['image_path']

        # Display image
        try:
            img = Image.open(img_path).convert('RGB')
            img_display = img.copy()
            img_display.thumbnail((480, 480), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img_display)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
        except Exception as e:
            self.gt_label.configure(text=f"Error: {e}")
            return

        # Run inference
        with torch.no_grad():
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            query = self.model(img_t).cpu().numpy()[0]

        # GPS filtering
        plat, plon = row['photo_lat'], row['photo_lon']
        candidates = []
        for key, (tlat, tlon) in self.tree_coords.items():
            if key not in self.tree_prototypes:
                continue
            dist = self._haversine(plat, plon, tlat, tlon)
            if dist <= self.radius_m:
                best_sim = max(float(np.dot(query, p)) for p in self.tree_prototypes[key])
                candidates.append((key, best_sim, dist))

        if not candidates:
            # Fallback to address matching
            addr = true_key.rsplit('|', 1)[0] if '|' in true_key else true_key
            for key in self.address_trees.get(addr, []):
                if key in self.tree_prototypes:
                    best_sim = max(float(np.dot(query, p)) for p in self.tree_prototypes[key])
                    candidates.append((key, best_sim, 0))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Update UI
        self.gt_label.configure(text=true_key)

        if candidates:
            pred_key = candidates[0][0]
            pred_sim = candidates[0][1]
            self.pred_label.configure(text=pred_key)
            self.sim_label.configure(text=f"{pred_sim:.4f}")
            self.cand_label.configure(text=f"{len(candidates)} trees within {self.radius_m}m")

            is_correct = pred_key == true_key
            self.tested += 1
            if is_correct:
                self.correct += 1
                self.result_label.configure(text="CORRECT", style="Correct.TLabel")
            else:
                self.result_label.configure(text="WRONG", style="Wrong.TLabel")

            # Top 3
            for i, lbl in enumerate(self.top3_labels):
                if i < len(candidates):
                    k, s, d = candidates[i]
                    marker = " <--" if k == true_key else ""
                    lbl.configure(text=f"{i+1}. {k} (sim={s:.4f}, {d:.0f}m){marker}")
                else:
                    lbl.configure(text=f"{i+1}. --")
        else:
            self.pred_label.configure(text="No candidates")
            self.sim_label.configure(text="--")
            self.cand_label.configure(text="0 trees")
            self.result_label.configure(text="NO MATCH", style="Wrong.TLabel")
            for lbl in self.top3_labels:
                lbl.configure(text="--")

        # Update stats
        acc = self.correct / self.tested * 100 if self.tested > 0 else 0
        self.stats_label.configure(text=f"Tested: {self.tested} | Correct: {self.correct} | Accuracy: {acc:.1f}%")

    def _toggle_auto(self):
        self.auto_running = not self.auto_running
        if self.auto_running:
            self.auto_btn.configure(text="Stop Auto")
            self._auto_next()
        else:
            self.auto_btn.configure(text="Auto Demo (3s)")

    def _auto_next(self):
        if self.auto_running:
            self._next_image()
            self.root.after(3000, self._auto_next)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\metric\best_model.pth')
    parser.add_argument('--data', default=r'E:\tree_id_2.0\data')
    parser.add_argument('--images', default=r'E:\tree_id_2.0\images')
    args = parser.parse_args()

    gui = TreeDemoGUI(args.model, args.data, args.images)
    gui.run()
