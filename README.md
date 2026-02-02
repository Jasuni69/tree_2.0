# Tree Identification System

Identify unique trees using GPS coordinates and visual matching. Submit photo with location, get back which tree it is.

## The Problem

- 3000+ unique trees spread across multiple areas
- Workers photograph trees and need to identify which tree it is
- Trees can be close together (10m apart)
- Phone GPS accuracy is ~3-5m (not precise enough alone)
- Ground truth coordinates exist but may have some errors

## The Solution

Multi-signal matching system:

```
Photo + GPS
    │
    ├── 1. Spatial Filter ──→ Find nearby candidates (fast)
    │
    ├── 2. Visual Match ───→ CNN compares to candidates
    │
    └── 3. Confidence ─────→ Return best match or top picks
```

## Features

- **Spatial Indexing**: KD-tree for fast nearest-neighbor queries
- **Visual Matching**: CNN embeddings for image similarity
- **Outlier Detection**: DBSCAN clustering to find bad coordinates
- **Interactive Map**: Browse trees by area, view images
- **Robust Matching**: Handles GPS uncertainty and close trees

## Quick Start

### Installation

```bash
cd tree_id_2.0
pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
scipy
scikit-learn
torch
torchvision
Pillow
folium
streamlit
streamlit-folium
matplotlib
```

## Usage

### 1. Prepare Data

Place your tree data in `data/` folder:

```
data/
├── trees.csv          # Coordinates and metadata
└── images/            # Tree images by ID
    ├── tree_001/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── tree_002/
    └── ...
```

Expected CSV format:
```csv
tree_id,lat,lon,address
1,40.7128,-74.0060,Area A - Block 1
2,40.7129,-74.0061,Area A - Block 1
3,40.7200,-74.0100,Area B - Block 1
```

### 2. Detect Outliers

Find potentially incorrect coordinates:

```python
from src.outliers import detect_outliers

outliers = detect_outliers('data/trees.csv')
# Outputs list of tree IDs with suspicious locations
```

Or visualize on map:

```python
from src.outliers import plot_outliers

plot_outliers('data/trees.csv', output='outliers_map.html')
```

### 3. Build Spatial Index

```python
from src.spatial import SpatialIndex

index = SpatialIndex('data/trees.csv')

# Find trees within 50 meters
candidates = index.query(lat=40.7128, lon=-74.0060, radius_m=50)
```

### 4. Precompute CNN Embeddings

Run once (takes time):

```python
from src.visual import compute_all_embeddings

compute_all_embeddings(
    image_dir='data/images',
    output='data/embeddings.pkl'
)
```

### 5. Match a Photo

```python
from src.pipeline import TreeMatcher

matcher = TreeMatcher(
    trees_csv='data/trees.csv',
    embeddings='data/embeddings.pkl'
)

result = matcher.identify('new_photo.jpg')
# Returns: {tree_id: 42, confidence: 0.87, candidates: [...]}
```

### 6. Interactive Map

```bash
streamlit run app/streamlit_app.py
```

Features:
- Select area/address from dropdown
- View all trees in that area on map
- Click tree to see all its images
- Overview map with clustering

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Photo                          │
│                 (with GPS metadata)                     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  EXIF Extraction                        │
│            Extract lat, lon from image                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Spatial Query                          │
│         KD-Tree: Find trees within 50m radius           │
│         3000 trees → ~5-15 candidates                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Visual Matching                        │
│         CNN embedding of new photo                      │
│         Cosine similarity to candidate embeddings       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 Confidence Scoring                      │
│         Combine: GPS distance + visual similarity       │
│         High confidence → return match                  │
│         Low confidence → return top candidates          │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      Output                             │
│         {tree_id, confidence, alternatives}             │
└─────────────────────────────────────────────────────────┘
```

## Handling Edge Cases

### Trees Very Close Together (10m)

GPS alone not enough. System uses:
1. Visual features (bark, shape, surroundings)
2. Sequence position (Tree 1 is first in row)
3. Relative position to landmarks

### Inaccurate Ground Truth

- DBSCAN detects coordinate outliers
- Flag low-confidence matches for review
- System learns from corrections over time

### Multiple Images per Tree

All images contribute to matching:
- Average embedding across images
- Or match against best single image

## Configuration

Key parameters to tune:

```python
# Spatial search radius (meters)
SEARCH_RADIUS = 50

# Minimum confidence to auto-accept match
CONFIDENCE_THRESHOLD = 0.7

# DBSCAN outlier detection
DBSCAN_EPS = 0.0003      # ~30m in degrees
DBSCAN_MIN_SAMPLES = 3
```

## Project Structure

```
tree_id_2.0/
├── data/
│   ├── trees.csv              # Tree coordinates
│   ├── embeddings.pkl         # Precomputed CNN vectors
│   └── images/                # Tree images
├── src/
│   ├── spatial.py             # Spatial indexing
│   ├── visual.py              # CNN embeddings
│   ├── pipeline.py            # Matching pipeline
│   ├── outliers.py            # Outlier detection
│   └── utils.py               # Helpers
├── app/
│   └── streamlit_app.py       # Interactive map
├── notebooks/                 # Exploration notebooks
├── tests/
├── requirements.txt
├── README.md
└── PLAN.md                    # Detailed project plan
```

## Roadmap

- [x] Project planning
- [ ] Data collection and formatting
- [ ] Spatial index implementation
- [ ] CNN embedding pipeline
- [ ] Outlier detection and visualization
- [ ] Interactive map application
- [ ] End-to-end matching pipeline
- [ ] REST API (future)
- [ ] Mobile integration (future)

## See Also

- [PLAN.md](PLAN.md) - Detailed implementation plan with phases and milestones
