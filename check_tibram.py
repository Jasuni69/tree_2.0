import pandas as pd
import re

# Read data
tibram = pd.read_excel('Tibram Trees.xlsx', skiprows=3, header=None)
tibram = tibram.dropna(axis=1, how='all')
tibram.columns = ['Location', 'TreeNumber', 'TreeCategory', 'Age', 'Latitude', 'Longitude', 'Polygon', 'DeadStatus']
tibram = tibram[~tibram['Location'].isin(['Location'])]

photos = pd.read_excel('data/photos_by_tree_with_outliers.xlsx')

def normalize_string(s):
    if pd.isna(s):
        return None
    # Strip, collapse whitespace
    return ' '.join(str(s).strip().split())

# Normalize locations
tibram['Location'] = tibram['Location'].apply(normalize_string)
photos['address'] = photos['address'].apply(normalize_string)

def normalize_tree_number(tree_str):
    if pd.isna(tree_str):
        return None
    numbers = re.findall(r'\d+', str(tree_str))
    if numbers:
        return int(numbers[0])
    return None

# Filter Tibram to trees with coordinates
tibram_trees = tibram[(tibram['Latitude'].notna()) & (tibram['TreeCategory'].isin(['Tree', 'Grey Tree']))].copy()
tibram_trees['tree_num_normalized'] = tibram_trees['TreeNumber'].apply(normalize_tree_number)

# Normalize photo tree numbers
photos['tree_num_normalized'] = photos['tree_number'].apply(normalize_tree_number)

# Create unique keys - ensure int type for tree numbers
tibram_trees['tree_num_int'] = tibram_trees['tree_num_normalized'].apply(lambda x: int(x) if pd.notna(x) else None)
photos['tree_num_int'] = photos['tree_num_normalized'].apply(lambda x: int(x) if pd.notna(x) else None)

tibram_trees['key'] = tibram_trees['Location'] + '|||' + tibram_trees['tree_num_int'].astype(str)
photos['key'] = photos['address'] + '|||' + photos['tree_num_int'].astype(str)

tibram_keys = set(tibram_trees['key'].unique())
photo_keys = set(photos['key'].unique())

overlap = tibram_keys.intersection(photo_keys)
print(f'=== Tree-level Matching ===')
print(f'Tibram trees with coords: {len(tibram_keys)}')
print(f'Photo trees: {len(photo_keys)}')
print(f'Matching trees: {len(overlap)} ({len(overlap)/len(photo_keys)*100:.1f}% of photo trees)')

# Show unmatched photo trees
unmatched = photo_keys - tibram_keys
print(f'\nUnmatched photo trees: {len(unmatched)}')
print(f'Sample unmatched:')
for key in list(unmatched)[:10]:
    print(f'  {key}')

# For matched trees, compare Tibram coords vs photo median
print('\n=== Coordinate Comparison (sample) ===')
matched_photos = photos[photos['key'].isin(overlap)].copy()
matched_tibram = tibram_trees[tibram_trees['key'].isin(overlap)].copy()

# Convert Tibram lat/lon to float
matched_tibram['Latitude'] = pd.to_numeric(matched_tibram['Latitude'], errors='coerce')
matched_tibram['Longitude'] = pd.to_numeric(matched_tibram['Longitude'], errors='coerce')

# Merge
merged = matched_photos.merge(
    matched_tibram[['key', 'Latitude', 'Longitude']],
    on='key',
    suffixes=('_photo', '_tibram')
)

# Calculate distance between photo median and Tibram coords
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Group by tree and compare
tree_comparison = merged.groupby('key').agg({
    'tree_median_lat': 'first',
    'tree_median_lon': 'first',
    'Latitude': 'first',
    'Longitude': 'first'
}).reset_index()

tree_comparison['distance_m'] = tree_comparison.apply(
    lambda r: haversine(r['tree_median_lat'], r['tree_median_lon'], r['Latitude'], r['Longitude']),
    axis=1
)

print(f'\nDistance between photo median and Tibram ground truth:')
print(f'  Mean: {tree_comparison["distance_m"].mean():.1f}m')
print(f'  Median: {tree_comparison["distance_m"].median():.1f}m')
print(f'  Max: {tree_comparison["distance_m"].max():.1f}m')
print(f'  < 5m: {(tree_comparison["distance_m"] < 5).sum()} ({(tree_comparison["distance_m"] < 5).sum()/len(tree_comparison)*100:.1f}%)')
print(f'  < 10m: {(tree_comparison["distance_m"] < 10).sum()} ({(tree_comparison["distance_m"] < 10).sum()/len(tree_comparison)*100:.1f}%)')
print(f'  < 20m: {(tree_comparison["distance_m"] < 20).sum()} ({(tree_comparison["distance_m"] < 20).sum()/len(tree_comparison)*100:.1f}%)')
print(f'  > 50m: {(tree_comparison["distance_m"] > 50).sum()} ({(tree_comparison["distance_m"] > 50).sum()/len(tree_comparison)*100:.1f}%)')
