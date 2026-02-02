# Tree Identification System - Project Plan

## Overview

System to identify unique trees using GPS coordinates and image matching. Workers submit photos with GPS metadata, system matches to known trees in database.

## Problem Statement

- ~3000 unique trees with ground truth coordinates (manually entered)
- Trees can be close together (10m spacing)
- GPS accuracy on phones ~3-5m (causes overlap)
- Need robust matching that handles coordinate uncertainty

---

## Phase 1: Data Preparation & Exploration

### 1.1 Data Collection
- [ ] Gather all existing tree data (coordinates, images, area/cluster info)
- [ ] Determine data format (CSV, database, image folders)
- [ ] Document coordinate storage (in image EXIF vs separate database)

### 1.2 Data Cleaning
- [ ] Validate coordinate formats (decimal degrees)
- [ ] Check for duplicate entries
- [ ] Flag suspicious coordinates (outliers)
- [ ] Verify image-to-coordinate mappings

### 1.3 Data Structure Design
```
tree_record = {
    "tree_id": int,
    "lat": float,
    "lon": float,
    "address/area": string,
    "cluster_id": int,
    "images": [list of image paths],
    "image_embeddings": [precomputed CNN vectors],
    "sequence_position": int (optional),
    "neighbors": {prev, next} (optional)
}
```

---

## Phase 2: Spatial Indexing

### 2.1 Build Spatial Index
Purpose: Fast filtering - don't compare against trees that are far away.

Options (pick one):
- **KD-Tree** (recommended for 3000 trees) - Simple, fast, in-memory
- **H3 Hexagonal Grid** - Good for larger scale, consistent neighbors
- **Geohash** - Simple string-based, easy database queries
- **PostGIS** - If using PostgreSQL, native spatial queries

### 2.2 Implementation
```python
# KD-Tree approach
from scipy.spatial import KDTree

coords = np.array([[t.lat, t.lon] for t in trees])
spatial_index = KDTree(coords)

def get_candidates(photo_lat, photo_lon, radius_meters=50):
    radius_deg = radius_meters / 111000
    indices = spatial_index.query_ball_point([photo_lat, photo_lon], radius_deg)
    return [trees[i] for i in indices]
```

### 2.3 Radius Tuning
- Start with 50m radius
- Adjust based on GPS accuracy and tree density
- Goal: Small enough to limit candidates, large enough to not miss correct tree

---

## Phase 3: Visual Matching (CNN)

### 3.1 Purpose
When multiple trees within GPS radius, use visual similarity to pick correct one.

### 3.2 Approach: Embedding-Based Matching
1. Use pretrained CNN (ResNet, EfficientNet) as feature extractor
2. Remove classification head, use embedding layer
3. Precompute embeddings for all stored tree images
4. At query time: compute embedding for new photo, cosine similarity to candidates

### 3.3 Implementation
```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
model.eval()

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img).squeeze().numpy()
    return embedding

def match_visual(query_embedding, candidate_embeddings):
    similarities = cosine_similarity([query_embedding], candidate_embeddings)
    return similarities[0]
```

### 3.4 Precompute Embeddings
- Run once on all 3000 trees
- Store in database or pickle file
- Update when new reference images added

---

## Phase 4: Matching Pipeline

### 4.1 Full Flow
```
Input: Photo with GPS metadata
           │
           ▼
┌──────────────────────────┐
│ 1. Extract GPS from EXIF │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ 2. Spatial Query         │
│    Find trees within 50m │
│    Result: N candidates  │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ 3. Visual Matching       │
│    CNN embedding compare │
│    Rank by similarity    │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ 4. Confidence Scoring    │
│    Combine GPS distance  │
│    + visual similarity   │
└──────────────────────────┘
           │
           ▼
Output: Best match + confidence
        Or: Top N candidates if uncertain
```

### 4.2 Confidence Scoring
```python
def calculate_confidence(gps_distance, visual_similarity, second_best_similarity):
    # GPS component (closer = better)
    gps_score = max(0, 1 - (gps_distance / 50))  # 0-1 scale

    # Visual component
    visual_score = visual_similarity  # Already 0-1

    # Margin - how much better than second choice
    margin = visual_similarity - second_best_similarity

    # Combined
    confidence = (0.3 * gps_score) + (0.5 * visual_score) + (0.2 * margin)
    return confidence
```

### 4.3 Handle Uncertainty
- If confidence < 0.7: Return top 3 candidates for human review
- If confidence > 0.9: Auto-accept match
- Log all matches for later analysis

---

## Phase 5: Outlier Detection & Data Quality

### 5.1 DBSCAN Clustering
Detect trees with suspicious coordinates.

```python
from sklearn.cluster import DBSCAN

coords = df[['lat', 'lon']].values
clustering = DBSCAN(eps=0.0003, min_samples=3).fit(coords)
df['cluster'] = clustering.labels_

outliers = df[df['cluster'] == -1]  # -1 = outlier
```

### 5.2 Visualization
- Plot all trees on map
- Color by cluster
- Highlight outliers in red
- Export list for manual review

### 5.3 Per-Area Analysis
Run outlier detection per address/area for localized anomalies.

---

## Phase 6: Interactive Map Application

### 6.1 Features
- View all trees clustered by address
- Select cluster → see trees in that area
- Select tree → view all images
- Highlight selected tree on map
- Overview map with all clusters

### 6.2 Tech Stack
- **Streamlit** - Quick Python web app
- **Folium** - Interactive maps
- **streamlit-folium** - Bridge between them

### 6.3 Screens
1. **Overview**: All clusters on map
2. **Cluster View**: Zoom to specific address, see all trees
3. **Tree Detail**: Selected tree info + all images

---

## Phase 7: API/Integration (Future)

### 7.1 REST API Endpoints
```
POST /identify
  Input: Image file
  Output: {tree_id, confidence, candidates[]}

GET /trees
  Output: All trees with coords

GET /trees/{id}
  Output: Tree details + images

GET /clusters
  Output: All clusters/addresses
```

### 7.2 Mobile Integration
- App sends photo with GPS
- Backend processes and returns match
- App shows result + confidence

---

## Tech Stack Summary

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Spatial Index | scipy KDTree or H3 |
| CNN | PyTorch + ResNet/EfficientNet |
| Database | SQLite (small) or PostgreSQL (scale) |
| Visualization | Matplotlib, Folium |
| Web App | Streamlit |
| API (future) | FastAPI |

---

## File Structure (Proposed)

```
tree_id_2.0/
├── data/
│   ├── trees.csv              # Tree coordinates and metadata
│   ├── embeddings.pkl         # Precomputed CNN embeddings
│   └── images/                # Tree images organized by ID
│       ├── tree_001/
│       ├── tree_002/
│       └── ...
├── src/
│   ├── __init__.py
│   ├── spatial.py             # Spatial indexing and queries
│   ├── visual.py              # CNN embedding and matching
│   ├── pipeline.py            # Full matching pipeline
│   ├── outliers.py            # Outlier detection
│   └── utils.py               # EXIF extraction, helpers
├── app/
│   └── streamlit_app.py       # Interactive map application
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_outlier_analysis.ipynb
│   └── 03_matching_tests.ipynb
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── PLAN.md
```

---

## Milestones

| Phase | Deliverable | Dependencies |
|-------|-------------|--------------|
| 1 | Clean dataset, documented format | Coordinate data |
| 2 | Working spatial index | Phase 1 |
| 3 | CNN embeddings for all trees | Phase 1 + images |
| 4 | End-to-end matching pipeline | Phases 2, 3 |
| 5 | Outlier report + visualizations | Phase 1 |
| 6 | Interactive map app | Phases 1, 5 |
| 7 | REST API | Phase 4 |

---

## Open Questions

1. **Data format**: How are coordinates stored? CSV? Database? Image EXIF?
2. **Image organization**: How are images linked to tree IDs?
3. **Areas/Clusters**: Is there existing address/area grouping?
4. **Sequence info**: Do we have tree order within rows?
5. **Update frequency**: How often do new trees/images get added?
6. **Deployment**: Local tool or cloud service?

---

## Next Steps

1. Get sample of coordinate data
2. Determine data format and structure
3. Run initial outlier detection
4. Build prototype spatial index
5. Test matching on known examples
