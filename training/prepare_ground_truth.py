"""
Prepare training data with Tibram ground truth coordinates.

Merges photo data with Tibram verified tree positions.
Filters to only trees with confirmed ground truth.
"""

import pandas as pd
import re
from pathlib import Path


def normalize_address(s):
    """Normalize address - strip and collapse whitespace."""
    if pd.isna(s):
        return ''
    return ' '.join(str(s).strip().split())


def normalize_tree_number(tree_str):
    """Extract number from tree names like 'Träd 1', 'Tree1'."""
    if pd.isna(tree_str):
        return None
    numbers = re.findall(r'\d+', str(tree_str))
    if numbers:
        return int(numbers[0])
    return None


def load_tibram_ground_truth(tibram_path):
    """Load and process Tibram ground truth data."""
    tibram = pd.read_excel(tibram_path, skiprows=3, header=None)
    tibram = tibram.dropna(axis=1, how='all')
    tibram.columns = ['Location', 'TreeNumber', 'TreeCategory', 'Age',
                      'Latitude', 'Longitude', 'Polygon', 'DeadStatus']
    tibram = tibram[~tibram['Location'].isin(['Location'])]

    # Filter to trees with coordinates
    tibram = tibram[(tibram['Latitude'].notna()) &
                    (tibram['TreeCategory'].isin(['Tree', 'Grey Tree']))]

    # Normalize
    tibram['loc_norm'] = tibram['Location'].apply(normalize_address)
    tibram['tree_num'] = tibram['TreeNumber'].apply(normalize_tree_number)
    tibram['gt_lat'] = pd.to_numeric(tibram['Latitude'], errors='coerce')
    tibram['gt_lon'] = pd.to_numeric(tibram['Longitude'], errors='coerce')

    # Create lookup key - handle NaN
    tibram = tibram[tibram['tree_num'].notna()]
    tibram['key'] = tibram['loc_norm'] + '|' + tibram['tree_num'].astype(int).astype(str)

    # Keep only needed columns
    tibram_clean = tibram[['key', 'loc_norm', 'tree_num', 'gt_lat', 'gt_lon', 'DeadStatus']].copy()
    tibram_clean = tibram_clean.drop_duplicates(subset='key')

    return tibram_clean


def prepare_training_data(data_dir, outlier_threshold_m=15):
    """
    Prepare training data with Tibram ground truth.

    Returns DataFrame with:
    - image_path, task_id
    - address, tree_number (original)
    - photo_lat, photo_lon (from EXIF)
    - gt_lat, gt_lon (Tibram ground truth)
    - distance_from_gt (photo distance from ground truth)
    """
    data_dir = Path(data_dir)

    # Load photo data
    photos = pd.read_excel(data_dir / 'photos_by_tree_with_outliers.xlsx')
    print(f"Total photos: {len(photos)}")

    # Normalize for matching
    photos['loc_norm'] = photos['address'].apply(normalize_address)
    photos['tree_num'] = photos['tree_number'].apply(normalize_tree_number)
    photos = photos[photos['tree_num'].notna()]
    photos['key'] = photos['loc_norm'] + '|' + photos['tree_num'].astype(int).astype(str)

    # Load Tibram ground truth
    tibram = load_tibram_ground_truth(data_dir.parent / 'Tibram Trees.xlsx')
    print(f"Tibram ground truth entries: {len(tibram)}")

    # Merge
    merged = photos.merge(tibram[['key', 'gt_lat', 'gt_lon', 'DeadStatus']],
                          on='key', how='inner')
    print(f"Photos with ground truth: {len(merged)}")

    # Calculate distance from ground truth
    import math
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    merged['distance_from_gt'] = merged.apply(
        lambda r: haversine(r['photo_lat'], r['photo_lon'], r['gt_lat'], r['gt_lon']),
        axis=1
    )

    # Filter outliers (photos too far from ground truth)
    if outlier_threshold_m:
        before = len(merged)
        merged = merged[merged['distance_from_gt'] <= outlier_threshold_m]
        print(f"After filtering >{outlier_threshold_m}m from GT: {len(merged)} ({len(merged)/before*100:.1f}%)")

    # Filter dead trees
    dead_count = (merged['DeadStatus'] == 'Marked as Dead').sum()
    merged = merged[merged['DeadStatus'] != 'Marked as Dead']
    print(f"Removed {dead_count} photos of dead trees")

    # Stats
    unique_trees = merged.groupby('key').ngroups
    print(f"\nFinal dataset:")
    print(f"  Photos: {len(merged)}")
    print(f"  Unique trees: {unique_trees}")
    print(f"  Avg photos per tree: {len(merged)/unique_trees:.1f}")

    # Distance stats
    print(f"\nPhoto distance from ground truth:")
    print(f"  Mean: {merged['distance_from_gt'].mean():.1f}m")
    print(f"  Median: {merged['distance_from_gt'].median():.1f}m")
    print(f"  Max: {merged['distance_from_gt'].max():.1f}m")

    return merged


def create_label_encoder(df):
    """Create mapping from key to class_id."""
    unique_trees = df['key'].unique()
    encoder = {key: i for i, key in enumerate(sorted(unique_trees))}
    decoder = {i: key for key, i in encoder.items()}
    return encoder, decoder


if __name__ == '__main__':
    DATA_DIR = Path(r'E:\tree_id_2.0\data')

    # Prepare data
    df = prepare_training_data(DATA_DIR, outlier_threshold_m=15)

    # Create encoder
    encoder, decoder = create_label_encoder(df)
    print(f"\nClasses: {len(encoder)}")

    # Save
    output_path = DATA_DIR / 'training_data_with_ground_truth.xlsx'
    df.to_excel(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Save encoder
    import json
    with open(DATA_DIR / 'label_encoder_gt.json', 'w') as f:
        json.dump(encoder, f)
    print(f"Saved encoder to {DATA_DIR / 'label_encoder_gt.json'}")
