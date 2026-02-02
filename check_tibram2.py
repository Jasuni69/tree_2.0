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

# Normalize
tibram['loc_norm'] = tibram['Location'].apply(norm_str)
photos['loc_norm'] = photos['address'].apply(norm_str)

tibram['tree_num'] = tibram['TreeNumber'].apply(norm_tree_num)
photos['tree_num'] = photos['tree_number'].apply(norm_tree_num)

# Filter tibram to trees with coordinates
tibram_trees = tibram[(tibram['Latitude'].notna()) & (tibram['TreeCategory'].isin(['Tree', 'Grey Tree']))].copy()

# Create keys
tibram_trees['key'] = tibram_trees['loc_norm'] + '|' + tibram_trees['tree_num'].astype(str)
photos['key'] = photos['loc_norm'] + '|' + photos['tree_num'].astype(str)

print("Sample Tibram keys:")
for k in tibram_trees['key'].head(10):
    print(f"  '{k}'")

print("\nSample Photo keys:")
for k in photos['key'].head(10):
    print(f"  '{k}'")

tibram_keys = set(tibram_trees['key'].unique())
photo_keys = set(photos['key'].unique())

overlap = tibram_keys.intersection(photo_keys)
print(f"\n=== Matching ===")
print(f"Tibram tree keys: {len(tibram_keys)}")
print(f"Photo tree keys: {len(photo_keys)}")
print(f"Overlap: {len(overlap)} ({len(overlap)/len(photo_keys)*100:.1f}%)")

if overlap:
    print(f"\nSample matching keys:")
    for k in list(overlap)[:10]:
        print(f"  {k}")
