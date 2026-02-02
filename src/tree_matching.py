"""
Tree matching logic using GPS centroids and probability scoring.

Key insight: Photo GPS = photographer position (0-5m from tree)
So tree centroid = median of all photo positions for that tree.
"""

import json
import statistics
import math
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TreeLocation:
    """Calculated location for a unique tree."""
    unique_id: str          # "Address | TreeNumber"
    address: str
    tree_number: str
    centroid_lat: float
    centroid_lon: float
    spread_meters: float    # Max distance between any two photos
    sample_count: int       # Number of GPS samples
    confidence: str         # HIGH, MEDIUM, LOW based on spread


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def calculate_spread(coords: list[tuple[float, float]]) -> float:
    """Calculate max distance between any two coordinates in meters."""
    if len(coords) < 2:
        return 0.0

    max_dist = 0.0
    for i, (lat1, lon1) in enumerate(coords):
        for lat2, lon2 in coords[i+1:]:
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            max_dist = max(max_dist, dist)

    return max_dist


def build_tree_locations(gps_results: list[dict]) -> list[TreeLocation]:
    """
    Build tree locations from GPS extraction results.
    Groups by unique tree (address + tree_number), calculates centroid and spread.
    """
    # Group by unique tree
    tree_data = {}

    for r in gps_results:
        if not r.get('gps'):
            continue

        unique_id = f"{r['address']} | {r['tree_number']}"

        if unique_id not in tree_data:
            tree_data[unique_id] = {
                'address': r['address'],
                'tree_number': r['tree_number'],
                'coords': []
            }

        tree_data[unique_id]['coords'].append((
            r['gps']['lat'],
            r['gps']['lon']
        ))

    # Calculate centroid and spread for each tree
    trees = []
    for unique_id, data in tree_data.items():
        coords = data['coords']

        if len(coords) == 0:
            continue

        # Centroid = median (more robust than mean)
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        centroid_lat = statistics.median(lats)
        centroid_lon = statistics.median(lons)

        # Calculate spread
        spread = calculate_spread(coords)

        # Confidence based on spread
        # 0-10m: HIGH (expected range for 0-5m photographer distance)
        # 10-20m: MEDIUM (some GPS drift or varied positions)
        # >20m: LOW (likely data issues)
        if spread <= 10:
            confidence = "HIGH"
        elif spread <= 20:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        trees.append(TreeLocation(
            unique_id=unique_id,
            address=data['address'],
            tree_number=data['tree_number'],
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            spread_meters=spread,
            sample_count=len(coords),
            confidence=confidence
        ))

    return trees


def find_candidates(
    photo_lat: float,
    photo_lon: float,
    trees: list[TreeLocation],
    radius_meters: float = 50.0
) -> list[TreeLocation]:
    """Find all trees within radius of photo GPS."""
    candidates = []

    for tree in trees:
        dist = haversine_distance(
            photo_lat, photo_lon,
            tree.centroid_lat, tree.centroid_lon
        )
        if dist <= radius_meters:
            candidates.append((tree, dist))

    # Sort by distance
    candidates.sort(key=lambda x: x[1])
    return candidates


def calculate_match_probability(
    photo_lat: float,
    photo_lon: float,
    tree: TreeLocation,
    visual_score: float = None  # 0-1, from CNN matching
) -> dict:
    """
    Calculate probability that photo matches this tree.

    Returns dict with:
    - probability: 0-1 overall match probability
    - distance: meters from photo to tree centroid
    - factors: breakdown of scoring factors
    """
    distance = haversine_distance(
        photo_lat, photo_lon,
        tree.centroid_lat, tree.centroid_lon
    )

    # Distance-based probability
    # 0-5m: HIGH (photographer right next to tree)
    # 5-10m: MEDIUM (edge of range, or between trees)
    # 10-15m: LOW (unlikely but possible with GPS drift)
    # >15m: VERY LOW
    if distance <= 5:
        dist_prob = 0.85
    elif distance <= 10:
        dist_prob = 0.50
    elif distance <= 15:
        dist_prob = 0.20
    else:
        dist_prob = 0.05

    # Confidence adjustment based on tree's data quality
    confidence_multiplier = {
        "HIGH": 1.0,
        "MEDIUM": 0.9,
        "LOW": 0.7
    }.get(tree.confidence, 0.8)

    # Base probability
    probability = dist_prob * confidence_multiplier

    # Visual matching boost (if available)
    visual_boost = 0.0
    if visual_score is not None:
        if visual_score > 0.8:
            visual_boost = 0.25
        elif visual_score > 0.6:
            visual_boost = 0.15
        elif visual_score > 0.4:
            visual_boost = 0.05

    probability = min(1.0, probability + visual_boost)

    return {
        'probability': round(probability, 3),
        'distance_meters': round(distance, 2),
        'factors': {
            'distance_prob': dist_prob,
            'confidence_multiplier': confidence_multiplier,
            'visual_boost': visual_boost,
            'tree_confidence': tree.confidence,
            'tree_samples': tree.sample_count
        }
    }


def match_photo_to_trees(
    photo_lat: float,
    photo_lon: float,
    trees: list[TreeLocation],
    address_filter: str = None,
    top_n: int = 5
) -> list[dict]:
    """
    Match a photo to candidate trees, returning ranked results.

    Args:
        photo_lat, photo_lon: GPS from photo EXIF
        trees: List of all known tree locations
        address_filter: Optional - only consider trees at this address
        top_n: Return top N matches

    Returns:
        List of matches with tree info and probability
    """
    # Filter by address if specified
    if address_filter:
        candidates = [t for t in trees if t.address == address_filter]
    else:
        candidates = trees

    # Find nearby trees (within 50m)
    nearby = find_candidates(photo_lat, photo_lon, candidates, radius_meters=50.0)

    if not nearby:
        return []

    # Calculate match probability for each
    results = []
    for tree, dist in nearby:
        match_info = calculate_match_probability(photo_lat, photo_lon, tree)
        results.append({
            'tree_id': tree.unique_id,
            'address': tree.address,
            'tree_number': tree.tree_number,
            'centroid': (tree.centroid_lat, tree.centroid_lon),
            'probability': match_info['probability'],
            'distance_meters': match_info['distance_meters'],
            'tree_confidence': tree.confidence,
            'tree_samples': tree.sample_count,
            'factors': match_info['factors']
        })

    # Sort by probability descending
    results.sort(key=lambda x: -x['probability'])

    return results[:top_n]


# CLI for testing
if __name__ == '__main__':
    DATA_DIR = Path(r'E:\tree_id_2.0\data')

    # Load GPS results
    results_path = DATA_DIR / 'gps_extraction_results.json'
    if not results_path.exists():
        print("No GPS results found. Run extract_gps.py first.")
        exit(1)

    print("Loading GPS results...")
    with open(results_path) as f:
        gps_results = json.load(f)

    print(f"Loaded {len(gps_results)} results")

    # Build tree locations
    print("Building tree locations...")
    trees = build_tree_locations(gps_results)
    print(f"Built {len(trees)} tree locations")

    # Stats
    high_conf = sum(1 for t in trees if t.confidence == "HIGH")
    med_conf = sum(1 for t in trees if t.confidence == "MEDIUM")
    low_conf = sum(1 for t in trees if t.confidence == "LOW")

    print(f"\nConfidence breakdown:")
    print(f"  HIGH: {high_conf}")
    print(f"  MEDIUM: {med_conf}")
    print(f"  LOW: {low_conf}")

    # Save tree locations
    output_path = DATA_DIR / 'tree_locations.json'
    tree_dicts = [
        {
            'unique_id': t.unique_id,
            'address': t.address,
            'tree_number': t.tree_number,
            'centroid_lat': t.centroid_lat,
            'centroid_lon': t.centroid_lon,
            'spread_meters': t.spread_meters,
            'sample_count': t.sample_count,
            'confidence': t.confidence
        }
        for t in trees
    ]

    with open(output_path, 'w') as f:
        json.dump(tree_dicts, f, indent=2)

    print(f"\nSaved tree locations to {output_path}")

    # Example matching
    print("\n--- Example Match ---")
    if trees:
        # Use first tree's centroid as test photo location
        test_tree = trees[0]
        print(f"Test: Photo at {test_tree.address}")

        matches = match_photo_to_trees(
            test_tree.centroid_lat,
            test_tree.centroid_lon,
            trees,
            address_filter=test_tree.address
        )

        for i, m in enumerate(matches[:3], 1):
            print(f"  {i}. {m['tree_number']}: {m['probability']*100:.0f}% ({m['distance_meters']:.1f}m)")
