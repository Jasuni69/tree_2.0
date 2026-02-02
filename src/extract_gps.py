"""
Extract GPS coordinates from tree images on D: drive.
Links to Excel data and builds coordinate database.
"""

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
import json
from tqdm import tqdm
import logging
from datetime import datetime

# Config
EXCEL_PATH = Path(r'E:\tree_id_new\data\excel_files\Tasks 2023-2025.xlsx')
IMAGE_BASE_PATH = Path(r'D:\Task')
OUTPUT_DIR = Path(r'E:\tree_id_2.0\data')
CHECKPOINT_EVERY = 500  # Save progress every N images

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_gps_from_image(img_path: Path) -> dict | None:
    """Extract GPS coordinates from image EXIF data."""
    try:
        img = Image.open(img_path)
        exif = img.getexif()

        if not exif:
            return None

        # Get GPS IFD
        gps_ifd = exif.get_ifd(0x8825)  # GPSInfo tag

        if not gps_ifd:
            return None

        gps_data = {}
        for tag_id, value in gps_ifd.items():
            tag_name = GPSTAGS.get(tag_id, tag_id)
            gps_data[tag_name] = value

        # Convert to decimal degrees
        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
            lat = gps_data['GPSLatitude']
            lon = gps_data['GPSLongitude']
            lat_ref = gps_data.get('GPSLatitudeRef', 'N')
            lon_ref = gps_data.get('GPSLongitudeRef', 'E')

            lat_decimal = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
            lon_decimal = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600

            if lat_ref == 'S':
                lat_decimal = -lat_decimal
            if lon_ref == 'W':
                lon_decimal = -lon_decimal

            result = {
                'lat': lat_decimal,
                'lon': lon_decimal,
                'altitude': float(gps_data.get('GPSAltitude', 0)) if gps_data.get('GPSAltitude') else None
            }

            # Get timestamp if available
            datetime_tag = exif.get(306)  # DateTime tag
            if datetime_tag:
                result['datetime'] = datetime_tag

            # Get device info
            make = exif.get(271)  # Make tag
            model = exif.get(272)  # Model tag
            if make:
                result['device_make'] = make
            if model:
                result['device_model'] = model

            return result

    except Exception as e:
        logger.debug(f"Error reading {img_path}: {e}")
        return None

    return None


def load_checkpoint(checkpoint_path: Path) -> set:
    """Load set of already processed task IDs."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_path: Path, processed_ids: set):
    """Save checkpoint of processed task IDs."""
    with open(checkpoint_path, 'w') as f:
        json.dump(list(processed_ids), f)


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Excel data
    logger.info(f"Loading Excel data from {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    logger.info(f"Loaded {len(df)} records")

    # Create unique tree identifier
    df['UniqueTreeId'] = df['AddressLine'] + ' | ' + df['TreeNumber'].astype(str)

    # Checkpoint handling
    checkpoint_path = OUTPUT_DIR / 'extraction_checkpoint.json'
    results_path = OUTPUT_DIR / 'gps_extraction_results.json'

    processed_ids = load_checkpoint(checkpoint_path)
    logger.info(f"Found {len(processed_ids)} already processed in checkpoint")

    # Load existing results if any
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = []

    # Filter to unprocessed
    df_remaining = df[~df['ParentTaskId'].astype(str).isin(processed_ids)]
    logger.info(f"Processing {len(df_remaining)} remaining records")

    # Stats
    stats = {
        'total': len(df_remaining),
        'processed': 0,
        'with_gps': 0,
        'no_gps': 0,
        'file_not_found': 0,
        'errors': 0
    }

    # Process images
    for idx, row in tqdm(df_remaining.iterrows(), total=len(df_remaining), desc="Extracting GPS"):
        task_id = str(row['ParentTaskId'])

        # Build image path from BeforePhoto column
        before_photo = row['BeforePhoto']
        if pd.isna(before_photo):
            stats['file_not_found'] += 1
            processed_ids.add(task_id)
            continue

        img_path = IMAGE_BASE_PATH / before_photo

        if not img_path.exists():
            stats['file_not_found'] += 1
            processed_ids.add(task_id)
            continue

        # Extract GPS
        gps_data = get_gps_from_image(img_path)

        result = {
            'task_id': task_id,
            'address': row['AddressLine'],
            'tree_number': row['TreeNumber'],
            'unique_tree_id': row['UniqueTreeId'],
            'image_path': str(before_photo),
            'gps': gps_data
        }

        results.append(result)

        if gps_data:
            stats['with_gps'] += 1
        else:
            stats['no_gps'] += 1

        stats['processed'] += 1
        processed_ids.add(task_id)

        # Checkpoint save
        if stats['processed'] % CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint_path, processed_ids)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Checkpoint saved. Stats: {stats}")

    # Final save
    save_checkpoint(checkpoint_path, processed_ids)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Extraction complete. Final stats: {stats}")

    # Create summary by unique tree
    logger.info("Building tree summary...")
    tree_coords = {}

    for r in results:
        tree_id = r['unique_tree_id']
        if tree_id not in tree_coords:
            tree_coords[tree_id] = {
                'address': r['address'],
                'tree_number': r['tree_number'],
                'coords': [],
                'task_count': 0
            }

        tree_coords[tree_id]['task_count'] += 1

        if r['gps']:
            tree_coords[tree_id]['coords'].append({
                'lat': r['gps']['lat'],
                'lon': r['gps']['lon'],
                'task_id': r['task_id']
            })

    # Calculate median coords for each tree
    tree_summary = []
    for tree_id, data in tree_coords.items():
        if data['coords']:
            lats = [c['lat'] for c in data['coords']]
            lons = [c['lon'] for c in data['coords']]

            # Use median to reduce outlier impact
            import statistics
            median_lat = statistics.median(lats)
            median_lon = statistics.median(lons)

            tree_summary.append({
                'unique_tree_id': tree_id,
                'address': data['address'],
                'tree_number': data['tree_number'],
                'median_lat': median_lat,
                'median_lon': median_lon,
                'coord_count': len(data['coords']),
                'task_count': data['task_count'],
                'lat_spread': max(lats) - min(lats) if len(lats) > 1 else 0,
                'lon_spread': max(lons) - min(lons) if len(lons) > 1 else 0
            })

    # Save tree summary
    summary_df = pd.DataFrame(tree_summary)
    summary_path = OUTPUT_DIR / 'trees_with_coords.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Tree summary saved to {summary_path}")
    logger.info(f"Trees with coordinates: {len(tree_summary)}")

    return results, tree_summary


if __name__ == '__main__':
    main()
