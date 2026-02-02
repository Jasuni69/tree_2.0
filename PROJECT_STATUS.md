# Tree Identification Project - Complete Status Report

**Last Updated:** January 2026
**Project Path:** `E:\tree_id_2.0`

---

## Project Overview

CNN-based tree identification system that matches photos of trees to known tree locations. Uses visual similarity (embeddings) combined with GPS proximity to identify which specific tree a photo belongs to.

### Key Stats
- **Total Trees:** 1,365 unique trees
- **Total Photos:** 11,780 photos
- **Best Accuracy:** 97.4% (proto_only method on cleaned data)
- **Model:** EfficientNet-B2 backbone (1408-dim features)

### Core Concept
Each tree has multiple photos. Given a new photo with GPS coordinates:
1. Find candidate trees within GPS radius (30m default)
2. Compare photo embedding to each candidate's stored embeddings
3. Predict the tree with highest similarity

---

## Directory Structure

```
E:\tree_id_2.0\
├── data/                    # Data files and metadata
├── images/                  # All tree photos (organized by date)
├── models/                  # Trained models and embeddings
├── training/                # Training scripts
├── inference/               # Prediction and evaluation scripts
├── tools/                   # Utility tools (manual review, etc.)
├── app/                     # Web app (map visualization)
└── src/                     # Core utilities
```

---

## Data Files (`data/`)

| File | Description |
|------|-------------|
| `training_data_with_ground_truth.xlsx` | Original labeled dataset (11,780 photos) |
| `training_data_cleaned.xlsx` | After auto-correction of 197 mislabeled photos |
| `training_data_ORIGINAL_BACKUP.xlsx` | Untouched backup of original |
| `training_data_working.xlsx` | Working copy used during cleaning |
| `flagged_for_review.xlsx` | 231 items flagged (197 auto-corrected + 34 uncertain) |
| `cleaning_report.html` | Visual HTML report of flagged items |
| `label_encoder_gt.json` | Tree key → integer label mapping |
| `trees_with_ground_truth.xlsx` | Tree-level data with coordinates |

### Data Schema
Key columns in training data:
- `image_path`: Relative path to image (e.g., "2023-02-12/Before/212055.png")
- `key`: Tree identifier format: `"address|tree_number"` (e.g., "Skrubba Malmväg|10")
- `photo_lat`, `photo_lon`: GPS where photo was taken
- `gt_lat`, `gt_lon`: Ground truth tree location

---

## Models (`models/`)

| File | Description |
|------|-------------|
| `model_seed42.pt` | **Primary trained model** - EfficientNet-B2, 30 epochs |
| `tree_embeddings_v2.pt` | **Current embeddings** - with prototypes and outlier filtering |
| `tree_embeddings.pt` | Old embeddings (simple averaging, no prototypes) |
| `label_encoder.json` | Label mapping |
| `checkpoint_seed42.pt` | Training checkpoint |

### Model Architecture
```python
class TreeClassifier(nn.Module):
    backbone: EfficientNet-B2 (timm)      # 1408-dim features
    classifier: Dropout(0.3) → Linear(1408, 1365)
```

### Training Config
- **Backbone:** efficientnet_b2
- **Image size:** 260×260
- **Batch size:** 128
- **Epochs:** 30
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealing
- **Val split:** 15%

### Embeddings V2 Config
- **Prototypes per tree:** 3 (k-means clustering)
- **Outlier threshold:** 0.5 (remove photos with similarity < 0.5 to mean)
- **Result:** ~2.5 prototypes per tree average

---

## Scripts Reference

### Training (`training/`)

| Script | Purpose | Command |
|--------|---------|---------|
| `train.py` | Train classification model | `python training/train.py --seed 42` |
| `dataset.py` | Dataset class and transforms | (imported by train.py) |
| `prepare_ground_truth.py` | Prepare labeled data | `python training/prepare_ground_truth.py` |

### Inference (`inference/`)

| Script | Purpose | Command |
|--------|---------|---------|
| `evaluate.py` | Evaluate all methods | `python inference/evaluate.py --embeddings models/tree_embeddings_v2.pt` |
| `compute_embeddings_v2.py` | Generate tree embeddings with prototypes | `python inference/compute_embeddings_v2.py` |
| `compute_embeddings.py` | Old simple embedding computation | (deprecated, use v2) |
| `clean_data.py` | Find/fix mislabeled photos | `python inference/clean_data.py` |
| `generate_report.py` | HTML report with predictions | `python inference/generate_report.py` |
| `predict.py` | Single image prediction | `python inference/predict.py --image path/to/image.jpg` |

### Tools (`tools/`)

| Script | Purpose | Command |
|--------|---------|---------|
| `manual_review.py` | Pygame UI for reviewing uncertain photos | `python tools/manual_review.py` |

---

## Evaluation Methods Compared

| Method | Accuracy | Description |
|--------|----------|-------------|
| **proto_only** | **97.4%** | Best. Max similarity to any prototype within same address |
| gps_cnn_top3 | 95.7% | GPS filter + CNN top-3 contains correct |
| proto_hybrid_0.5 | 93.7% | 50% distance + 50% proto similarity |
| ensemble_hybrid | 92.9% | Ensemble of methods |
| emb_only | 92.8% | Pure embedding similarity (single mean) |
| hybrid_0.5 | 90.7% | 50% distance + 50% CNN logits |
| gps_cnn | 88.7% | GPS filter + CNN classification |
| cnn_only | 83.4% | Pure CNN classification (no GPS) |
| gps_only | 67.4% | Nearest tree by GPS only |

### Key Insight
Proto-based similarity (97.4%) beats CNN classification (83.4%) significantly. The model's feature embeddings are more discriminative than its classification head.

---

## Data Cleaning Results

Ran `clean_data.py` to find mislabeled photos:

| Status | Count | Percent |
|--------|-------|---------|
| Correct | 11,549 | 98.0% |
| Mislabeled (auto-corrected) | 197 | 1.7% |
| Uncertain (needs review) | 34 | 0.3% |

**Important constraint:** Cleaning tool only reassigns tree numbers within the SAME ADDRESS. Key format is `"address|tree_number"`, so corrections only happen between trees at same address.

After cleaning: accuracy improved from 94.8% → 97.4%

---

## How Prediction Works

### Inference Pipeline (proto_only method)
```
1. Load image → extract 1408-dim embedding via backbone
2. Normalize embedding to unit vector
3. Find candidate trees within same address
4. For each candidate:
   - Get its prototypes (1-3 cluster centers from training photos)
   - Compute max similarity to any prototype
5. Return tree with highest similarity
```

### Why Prototypes?
Trees have visual variation (seasons, angles, lighting). K-means clustering captures 2-3 "views" of each tree. Matching against the closest prototype handles this variation better than a single averaged embedding.

---

## Common Commands

```bash
# Activate environment
cd E:\tree_id_2.0
.\venv\Scripts\activate

# Train model
python training/train.py --seed 42 --epochs 30 --backbone efficientnet_b2

# Compute embeddings (after training)
python inference/compute_embeddings_v2.py

# Evaluate accuracy
python inference/evaluate.py --embeddings models/tree_embeddings_v2.pt

# Evaluate on cleaned data
python inference/evaluate.py --embeddings models/tree_embeddings_v2.pt --input training_data_cleaned.xlsx

# Clean data (find mislabeled photos)
python inference/clean_data.py --threshold 0.7

# Manual review of uncertain photos
python tools/manual_review.py

# Generate HTML prediction report
python inference/generate_report.py --sample 100
```

---

## Key Design Decisions

### 1. Address-Constrained Matching
Trees are only confused with other trees at the same address. The key format `"address|tree_number"` enforces this - a photo at "Main St|3" can only be reassigned to "Main St|1", "Main St|2", etc., never to a different street.

### 2. Prototype-Based Matching vs Classification
The model was trained for classification but embeddings work better:
- Classification: 83.4% (model predicts class directly)
- Embedding similarity: 97.4% (compare feature vectors)

This is because the embedding space captures visual similarity better than the classification boundary.

### 3. Outlier Filtering in Embeddings
Before computing prototypes, photos with embedding similarity < 0.5 to the tree's mean are removed. This filters mislabeled or unusual photos from corrupting the embeddings.

### 4. GPS Radius Not Used in Best Method
The proto_only method doesn't use GPS radius filtering - it only uses address matching. GPS proximity within an address is less reliable than visual similarity.

---

## Files to Read for Quick Context

1. **This file** (`PROJECT_STATUS.md`) - overall status
2. **`inference/evaluate.py`** - all evaluation methods
3. **`inference/clean_data.py`** - data cleaning logic
4. **`training/train.py`** - model architecture and training
5. **`inference/compute_embeddings_v2.py`** - prototype computation

---

## Current State (January 2026)

### Completed
- [x] Model trained (EfficientNet-B2, 30 epochs)
- [x] Embeddings computed with prototypes (v2)
- [x] Evaluation framework with multiple methods
- [x] Data cleaning tool built and run
- [x] 197 mislabeled photos auto-corrected
- [x] Manual review tool built for 34 uncertain photos
- [x] Accuracy at 97.4% on cleaned data

### Pending
- [ ] Manual review of 34 uncertain photos (run `python tools/manual_review.py`)
- [ ] Optionally retrain on cleaned data to see if accuracy improves further
- [ ] Potentially regenerate embeddings using cleaned data

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in evaluation:
```python
# In evaluate.py, process images one at a time (current behavior)
# For training, reduce --batch_size to 64 or 32
```

### Slow Evaluation
Evaluation processes all 11,780 photos individually. Takes ~1 hour on GPU.

### Missing Embeddings
If `tree_embeddings_v2.pt` doesn't exist:
```bash
python inference/compute_embeddings_v2.py
```

### Wrong Accuracy Numbers
Make sure using correct input file:
- Original data: `--input training_data_with_ground_truth.xlsx` (94.8%)
- Cleaned data: `--input training_data_cleaned.xlsx` (97.4%)

---

## Contact / Notes

This project identifies individual trees from photos for urban forestry management. Each tree has a unique key combining street address and tree number. The goal is to automatically match field photos to the correct tree in the database.
