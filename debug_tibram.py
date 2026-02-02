import pandas as pd
import re

tibram = pd.read_excel('Tibram Trees.xlsx', skiprows=3, header=None)
tibram = tibram.dropna(axis=1, how='all')
tibram.columns = ['Location', 'TreeNumber', 'TreeCategory', 'Age', 'Latitude', 'Longitude', 'Polygon', 'DeadStatus']
tibram = tibram[~tibram['Location'].isin(['Location'])]

photos = pd.read_excel('data/photos_by_tree_with_outliers.xlsx')

def norm_str(s):
    if pd.isna(s):
        return None
    return ' '.join(str(s).strip().split())

tibram['Location'] = tibram['Location'].apply(norm_str)
photos['address'] = photos['address'].apply(norm_str)

# Check location overlap first
tibram_locs = set(tibram['Location'].dropna().unique())
photo_locs = set(photos['address'].dropna().unique())

loc_overlap = tibram_locs.intersection(photo_locs)
print(f"Location overlap: {len(loc_overlap)} / {len(photo_locs)}")

# Pick a location that exists in overlap
if loc_overlap:
    test_loc = list(loc_overlap)[0]
    print(f"\nTesting location: {test_loc}")

    tibram_test = tibram[tibram['Location'] == test_loc]
    photos_test = photos[photos['address'] == test_loc]

    print(f"Tibram rows: {len(tibram_test)}")
    print(f"Photo rows: {len(photos_test)}")

    print(f"\nTibram TreeNumbers:")
    for tn in tibram_test['TreeNumber'].unique()[:10]:
        print(f"  '{tn}'")

    print(f"\nPhoto tree_numbers:")
    for tn in photos_test['tree_number'].unique()[:10]:
        print(f"  '{tn}'")

    # Check normalized numbers
    def normalize_tree_number(tree_str):
        if pd.isna(tree_str):
            return None
        numbers = re.findall(r'\d+', str(tree_str))
        if numbers:
            return int(numbers[0])
        return None

    tibram_nums = set(tibram_test['TreeNumber'].apply(normalize_tree_number).dropna().unique())
    photo_nums = set(photos_test['tree_number'].apply(normalize_tree_number).dropna().unique())

    print(f"\nNormalized tree numbers:")
    print(f"  Tibram: {sorted(tibram_nums)[:15]}")
    print(f"  Photos: {sorted(photo_nums)[:15]}")
    print(f"  Overlap: {sorted(tibram_nums.intersection(photo_nums))[:15]}")
