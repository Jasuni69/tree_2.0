import pandas as pd
import re
import math

# Read data
tibram = pd.read_excel('Tibram Trees.xlsx', skiprows=3, header=None)
tibram = tibram.dropna(axis=1, how='all')
tibram.columns = ['Location', 'TreeNumber', 'TreeCategory', 'Age', 'Latitude', 'Longitude', 'Polygon', 'DeadStatus']
tibram = tibram[~tibram['Location'].isin(['Location'])]

photos = pd.read_excel('data/photos_by_tree_with_outliers.xlsx')

def norm_str(s):
    if pd.isna(s):
        return ''
    return ' '.join(str(s).strip().split())

def norm_tree_num(tree_str):
    if pd.isna(tree_str):
        return -1
    numbers = re.findall(r'\d+', str(tree_str))
    if numbers:
        return int(numbers[0])
    return -1

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Normalize
tibram['loc_norm'] = tibram['Location'].apply(norm_str)
photos['loc_norm'] = photos['address'].apply(norm_str)

tibram['tree_num'] = tibram['TreeNumber'].apply(norm_tree_num)
photos['tree_num'] = photos['tree_number'].apply(norm_tree_num)

# Filter tibram to trees with coordinates
tibram_trees = tibram[(tibram['Latitude'].notna()) & (tibram['TreeCategory'].isin(['Tree', 'Grey Tree']))].copy()

# Convert lat/lon to float
tibram_trees['lat'] = pd.to_numeric(tibram_trees['Latitude'], errors='coerce')
tibram_trees['lon'] = pd.to_numeric(tibram_trees['Longitude'], errors='coerce')

# Create keys
tibram_trees['key'] = tibram_trees['loc_norm'] + '|' + tibram_trees['tree_num'].astype(str)
photos['key'] = photos['loc_norm'] + '|' + photos['tree_num'].astype(str)

# Get unique photo trees with their median coords
photo_trees = photos.groupby('key').agg({
    'tree_median_lat': 'first',
    'tree_median_lon': 'first',
    'address': 'first',
    'tree_number': 'first'
}).reset_index()

# Get tibram trees (take first if duplicates)
tibram_unique = tibram_trees.groupby('key').agg({
    'lat': 'first',
    'lon': 'first',
    'Location': 'first',
    'TreeNumber': 'first'
}).reset_index()

# Merge
merged = photo_trees.merge(tibram_unique, on='key', how='inner')

print(f"=== Coordinate Comparison ===")
print(f"Matched trees: {len(merged)}")

# Calculate distances
merged['distance_m'] = merged.apply(
    lambda r: haversine(r['tree_median_lat'], r['tree_median_lon'], r['lat'], r['lon']),
    axis=1
)

print(f"\nDistance between photo median and Tibram ground truth:")
print(f"  Mean:   {merged['distance_m'].mean():.1f}m")
print(f"  Median: {merged['distance_m'].median():.1f}m")
print(f"  Std:    {merged['distance_m'].std():.1f}m")
print(f"  Min:    {merged['distance_m'].min():.1f}m")
print(f"  Max:    {merged['distance_m'].max():.1f}m")

print(f"\nDistance distribution:")
print(f"  < 5m:   {(merged['distance_m'] < 5).sum()} ({(merged['distance_m'] < 5).sum()/len(merged)*100:.1f}%)")
print(f"  < 10m:  {(merged['distance_m'] < 10).sum()} ({(merged['distance_m'] < 10).sum()/len(merged)*100:.1f}%)")
print(f"  < 15m:  {(merged['distance_m'] < 15).sum()} ({(merged['distance_m'] < 15).sum()/len(merged)*100:.1f}%)")
print(f"  < 20m:  {(merged['distance_m'] < 20).sum()} ({(merged['distance_m'] < 20).sum()/len(merged)*100:.1f}%)")
print(f"  > 50m:  {(merged['distance_m'] > 50).sum()} ({(merged['distance_m'] > 50).sum()/len(merged)*100:.1f}%)")
print(f"  > 100m: {(merged['distance_m'] > 100).sum()} ({(merged['distance_m'] > 100).sum()/len(merged)*100:.1f}%)")

# Show worst matches
print(f"\n=== Worst matches (largest distance) ===")
worst = merged.nlargest(10, 'distance_m')
for _, row in worst.iterrows():
    print(f"  {row['address']} - {row['tree_number']}: {row['distance_m']:.1f}m")

# Save merged data for future use
merged.to_excel('data/trees_with_ground_truth.xlsx', index=False)
print(f"\nSaved matched data to data/trees_with_ground_truth.xlsx")
