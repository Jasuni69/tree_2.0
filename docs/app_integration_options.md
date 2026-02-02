# Tree ID App Integration Options

## Current System Performance

| Stage | Accuracy | Method |
|---|---|---|
| Model only (no GPS) | 72.7% | Prototype matching, address-scoped |
| Model + GPS 30m | 78.3% | GPS filters candidates, model ranks |
| Model + GPS 15m + TTA | 83.8% | Tight GPS + 7-view test-time augmentation |
| Top-3 accuracy | 94.6% | Correct tree in top 3 candidates |

The remaining 16.2% errors are same-address trees that are visually indistinguishable in close-up photos with inaccurate GPS. No amount of model training fixes this — it's a data problem.

---

## Option A: Pre-Selection (Recommended)

**How it works:** Worker selects which tree they're about to photograph before taking the image.

### App Flow
1. Worker arrives at address
2. App shows map/list of trees at that address (from database)
3. Worker taps the tree they're standing at
4. Worker takes photo
5. App sends: `{ image, selected_tree_id, gps_coords, timestamp }`

### Backend Integration
```
POST /api/photo
{
  "image": <file>,
  "selected_tree_id": "address|14",
  "photo_lat": 59.332,
  "photo_lon": 18.065
}
```

Backend runs the model as **verification** (not identification):
- Compute embedding of uploaded photo
- Compare against prototypes for the selected tree
- If cosine similarity > threshold (e.g. 0.7) → accept
- If below threshold → flag for review

### Pros
- Near 100% accuracy when workers select correctly
- Model becomes a sanity check, not the primary ID
- Simpler inference (compare against 1 tree, not all candidates)
- Faster response time

### Cons
- Extra step for the worker (tap before photo)
- Workers can select wrong tree — identical trees on same street
- Requires tree database in the app

---

## Option B: Pre-Selection + GPS Sanity Check (Recommended+)

**Extension of Option A.** Adds a GPS distance check to catch wrong selections.

### App Flow
1. Worker arrives at address
2. App waits for GPS lock (accuracy < 10m)
3. App shows map with trees and worker's position
4. Worker taps tree → app checks distance to selected tree
5. If worker GPS is closer to a different tree → prompt: "You're closer to tree #3. Did you mean that one?"
6. Worker confirms or changes selection
7. Photo taken and sent with confirmed tree ID

### Backend Integration
Same as Option A, plus:
```python
def validate_selection(photo_lat, photo_lon, selected_tree, nearby_trees):
    selected_dist = haversine(photo_lat, photo_lon, selected_tree.lat, selected_tree.lon)
    closest_tree = min(nearby_trees, key=lambda t: haversine(photo_lat, photo_lon, t.lat, t.lon))

    if closest_tree.id != selected_tree.id and selected_dist > closest_tree.dist + 3.0:
        return "mismatch_warning"
    return "ok"
```

### Failure Condition
Only fails when ALL of these are true simultaneously:
- Trees are visually identical (model can't distinguish)
- GPS is inaccurate (can't distinguish by position)
- Worker selects wrong tree (human error)

This is a very narrow failure window.

### Pros
- Catches most wrong selections via GPS
- Worker still has final say
- Three independent signals: human selection, GPS, model

### Cons
- GPS must stabilize first (may slow workflow)
- Doesn't help when trees are < 3m apart with bad GPS

---

## Option C: Top-3 Confirmation (No Pre-Selection)

**How it works:** System identifies the tree, shows top candidates when uncertain.

### App Flow
1. Worker takes photo
2. App sends image + GPS to backend
3. Backend runs model (TTA + GPS filtering)
4. If confidence high (margin > 0.05) → auto-accept, show green checkmark
5. If confidence low → show top 3 candidate trees with reference photos
6. Worker taps the correct one

### Backend Integration
```
POST /api/identify
{
  "image": <file>,
  "photo_lat": 59.332,
  "photo_lon": 18.065
}

Response (confident):
{
  "status": "confirmed",
  "tree_id": "address|14",
  "confidence": 0.92
}

Response (uncertain):
{
  "status": "needs_confirmation",
  "candidates": [
    {"tree_id": "address|14", "score": 0.85, "reference_image": "url"},
    {"tree_id": "address|15", "score": 0.84, "reference_image": "url"},
    {"tree_id": "address|16", "score": 0.79, "reference_image": "url"}
  ]
}
```

### Threshold Tuning
| Confidence Threshold | Auto-accept Rate | Prompts Shown | Effective Accuracy |
|---|---|---|---|
| Low (0.01) | ~84% | ~16% | ~95% |
| Medium (0.03) | ~70% | ~30% | ~97% |
| High (0.05) | ~50% | ~50% | ~99% |

These are estimates. Exact numbers need calibration on production data.

### Pros
- No extra step for most photos (83.8% auto-accepted)
- Worker only prompted when model is genuinely unsure
- No changes to capture workflow

### Cons
- Requires reference images served to the app
- Worker must visually compare trees (hard if they're identical)
- Heavier backend (TTA = 7 forward passes per photo)
- Still relies on model for initial ranking

---

## Option D: Sequential Capture (Workflow Change)

**How it works:** Workers photograph trees in order along a route, and the app tracks which tree is next.

### App Flow
1. Backend assigns a route: tree #1 → #2 → #3 → ... along the street
2. App shows "Next: Tree #7 at Drottningholmsvägen" with map pin
3. Worker walks to tree, takes photo
4. App auto-advances to next tree
5. If worker skips or goes out of order → app prompts to confirm

### Backend Integration
```
POST /api/route/{route_id}/photo
{
  "image": <file>,
  "expected_tree_id": "address|7",
  "sequence_number": 7,
  "photo_lat": 59.332,
  "photo_lon": 18.065
}
```

### Pros
- Eliminates selection entirely — app knows which tree is next
- GPS drift less of an issue (sequential order is a strong signal)
- Model only needed for verification

### Cons
- Rigid workflow — workers can't skip or reorder
- Requires route planning infrastructure
- Breaks if worker misses a tree and doesn't notice

---

## Comparison

| | Pre-Selection (B) | Top-3 (C) | Sequential (D) |
|---|---|---|---|
| Extra worker effort | 1 tap before photo | Occasional 1 tap after | None (but rigid order) |
| App complexity | Medium | Medium | High |
| Backend complexity | Low | High (TTA inference) | Medium |
| Expected accuracy | ~98%+ | ~95-97% | ~99% |
| Works with identical trees | Mostly (GPS helps) | Poorly | Yes |
| Workflow disruption | Low | Low | High |

## Recommendation

**Option B (Pre-Selection + GPS Check)** is the best balance. It's simple to build, minimally disruptive to workers, and solves the fundamental problem: the app knows which tree the worker is at before the photo is taken.

The model (83.8% accuracy) runs as a background verification layer. It catches the rare case where a worker selects the wrong tree but the GPS didn't catch it.

For the ~1-2% of cases where trees are truly identical, GPS is bad, and the worker picks wrong — accept that as the error floor. No system can solve that without physically tagging the trees.

---

## Model Deployment Notes

The trained model is at: `E:\tree_id_2.0\models\metric_384\best_model.pth`

Inference requirements:
- **Input:** 384x384 RGB image, normalized (ImageNet stats)
- **Output:** 1024-dim L2-normalized embedding vector
- **Model:** ConvNeXt-Base backbone + GeM pooling + embedding head
- **GPU memory:** ~2GB for single image inference
- **Latency:** ~50ms per image on GPU (single forward pass, no TTA)
- **With TTA:** ~350ms (7 forward passes)

For production, TTA may not be needed if using Option B (pre-selection). Single forward pass for verification is sufficient and 7x faster.
