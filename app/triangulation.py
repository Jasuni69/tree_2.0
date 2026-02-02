"""
Tree Position Triangulation using GPS + Camera Direction.

Each photo is a ray from camera position in the direction it was facing.
Where rays intersect = estimated true tree position.
"""

import math
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
import pandas as pd
from collections import defaultdict


def extract_gps_with_direction(image_path):
    """Extract GPS position and camera direction from image EXIF."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None

        gps_info = None
        for tag_id, value in exif.items():
            if TAGS.get(tag_id) == 'GPSInfo':
                gps_info = value
                break

        if not gps_info:
            return None

        # Parse GPS data
        gps_data = {}
        for tag_id, value in gps_info.items():
            tag = GPSTAGS.get(tag_id, tag_id)
            gps_data[tag] = value

        # Get lat/lon
        if 'GPSLatitude' not in gps_data or 'GPSLongitude' not in gps_data:
            return None

        lat = gps_data['GPSLatitude']
        lon = gps_data['GPSLongitude']

        # Convert to decimal degrees
        lat_decimal = lat[0] + lat[1]/60 + lat[2]/3600
        lon_decimal = lon[0] + lon[1]/60 + lon[2]/3600

        if gps_data.get('GPSLatitudeRef') == 'S':
            lat_decimal = -lat_decimal
        if gps_data.get('GPSLongitudeRef') == 'W':
            lon_decimal = -lon_decimal

        # Get camera direction (degrees from true north)
        direction = gps_data.get('GPSImgDirection')
        if direction is None:
            direction = gps_data.get('GPSDestBearing')

        # Get GPS accuracy if available
        accuracy = gps_data.get('GPSHPositioningError')

        return {
            'lat': lat_decimal,
            'lon': lon_decimal,
            'direction': float(direction) if direction else None,
            'accuracy': float(accuracy) if accuracy else None
        }
    except Exception as e:
        return None


def ray_intersection(p1_lat, p1_lon, dir1, p2_lat, p2_lon, dir2):
    """
    Find intersection point of two rays.

    Each ray starts at (lat, lon) and goes in direction (degrees from north).
    Returns intersection point or None if rays are parallel.
    """
    # Convert to radians
    dir1_rad = math.radians(dir1)
    dir2_rad = math.radians(dir2)

    # Convert lat/lon to local meters (approximate)
    # At Stockholm latitude (~59°), 1 degree lat ≈ 111km, 1 degree lon ≈ 57km
    lat_scale = 111000  # meters per degree latitude
    lon_scale = 111000 * math.cos(math.radians((p1_lat + p2_lat) / 2))

    # Convert to local coordinates (meters)
    x1, y1 = 0, 0
    x2 = (p2_lon - p1_lon) * lon_scale
    y2 = (p2_lat - p1_lat) * lat_scale

    # Direction vectors (north = +y, east = +x)
    # Direction is degrees clockwise from north
    dx1 = math.sin(dir1_rad)
    dy1 = math.cos(dir1_rad)
    dx2 = math.sin(dir2_rad)
    dy2 = math.cos(dir2_rad)

    # Solve for intersection: p1 + t1*d1 = p2 + t2*d2
    # x1 + t1*dx1 = x2 + t2*dx2
    # y1 + t1*dy1 = y2 + t2*dy2

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-10:
        return None  # Parallel rays

    t1 = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom

    # Only consider forward intersections (t1 > 0)
    if t1 < 0:
        return None

    # Calculate intersection point in local coords
    ix = x1 + t1 * dx1
    iy = y1 + t1 * dy1

    # Convert back to lat/lon
    int_lon = p1_lon + ix / lon_scale
    int_lat = p1_lat + iy / lat_scale

    # Distance from each camera to intersection
    dist1 = t1  # Already in meters
    dist2 = math.sqrt((ix - x2)**2 + (iy - y2)**2)

    return {
        'lat': int_lat,
        'lon': int_lon,
        'dist_from_p1': dist1,
        'dist_from_p2': dist2
    }


def triangulate_tree_position(photos_with_direction, max_distance=50):
    """
    Estimate tree position from multiple photos using triangulation.

    Args:
        photos_with_direction: List of dicts with 'lat', 'lon', 'direction'
        max_distance: Ignore intersections further than this (meters)

    Returns:
        Dict with estimated position and confidence metrics
    """
    # Filter photos that have direction data
    valid_photos = [p for p in photos_with_direction if p.get('direction') is not None]

    if len(valid_photos) < 2:
        # Can't triangulate with less than 2 rays
        if len(valid_photos) == 1:
            return {
                'lat': valid_photos[0]['lat'],
                'lon': valid_photos[0]['lon'],
                'method': 'single_photo',
                'confidence': 'low',
                'intersection_count': 0
            }
        return None

    # Find all ray intersections
    intersections = []
    for i in range(len(valid_photos)):
        for j in range(i + 1, len(valid_photos)):
            p1 = valid_photos[i]
            p2 = valid_photos[j]

            intersection = ray_intersection(
                p1['lat'], p1['lon'], p1['direction'],
                p2['lat'], p2['lon'], p2['direction']
            )

            if intersection:
                # Filter out intersections too far from cameras
                if intersection['dist_from_p1'] < max_distance and intersection['dist_from_p2'] < max_distance:
                    intersections.append(intersection)

    if not intersections:
        # No valid intersections - fall back to median
        lats = [p['lat'] for p in valid_photos]
        lons = [p['lon'] for p in valid_photos]
        return {
            'lat': np.median(lats),
            'lon': np.median(lons),
            'method': 'median_fallback',
            'confidence': 'low',
            'intersection_count': 0
        }

    # Use median of intersection points (robust to outliers)
    int_lats = [i['lat'] for i in intersections]
    int_lons = [i['lon'] for i in intersections]

    est_lat = np.median(int_lats)
    est_lon = np.median(int_lons)

    # Calculate spread of intersections as confidence metric
    spread = np.std(int_lats) * 111000 + np.std(int_lons) * 57000  # rough meters

    confidence = 'high' if spread < 5 else 'medium' if spread < 15 else 'low'

    return {
        'lat': est_lat,
        'lon': est_lon,
        'method': 'triangulation',
        'confidence': confidence,
        'intersection_count': len(intersections),
        'intersection_spread_m': spread
    }


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def test_triangulation_on_tree(address, tree_number, df_outliers, image_base):
    """Test triangulation on a specific tree and compare to median method."""

    # Get photos for this tree
    tree_photos = df_outliers[
        (df_outliers['address'] == address) &
        (df_outliers['tree_number'] == tree_number)
    ]

    if len(tree_photos) == 0:
        return None

    print(f"\nTree: {tree_number} at {address}")
    print(f"Photos: {len(tree_photos)}")

    # Extract GPS + direction from each photo
    photos_with_direction = []
    for _, row in tree_photos.iterrows():
        img_path = image_base / row['image_path']
        gps_data = extract_gps_with_direction(img_path)

        if gps_data:
            gps_data['image_path'] = row['image_path']
            photos_with_direction.append(gps_data)
            if gps_data['direction']:
                print(f"  {row['image_path']}: dir={gps_data['direction']:.1f}°, acc={gps_data.get('accuracy', 'N/A')}")

    photos_with_dir = [p for p in photos_with_direction if p.get('direction')]
    print(f"Photos with direction data: {len(photos_with_dir)}/{len(photos_with_direction)}")

    if len(photos_with_dir) < 2:
        print("  Not enough photos with direction for triangulation")
        return None

    # Triangulate
    result = triangulate_tree_position(photos_with_dir)

    # Compare to median method
    median_lat = tree_photos['tree_median_lat'].iloc[0]
    median_lon = tree_photos['tree_median_lon'].iloc[0]

    if result and result['method'] == 'triangulation':
        dist_from_median = haversine_distance(
            result['lat'], result['lon'],
            median_lat, median_lon
        )

        print(f"\nResults:")
        print(f"  Median position:       {median_lat:.6f}, {median_lon:.6f}")
        print(f"  Triangulated position: {result['lat']:.6f}, {result['lon']:.6f}")
        print(f"  Distance between:      {dist_from_median:.1f}m")
        print(f"  Intersections used:    {result['intersection_count']}")
        print(f"  Intersection spread:   {result.get('intersection_spread_m', 0):.1f}m")
        print(f"  Confidence:            {result['confidence']}")

        result['median_lat'] = median_lat
        result['median_lon'] = median_lon
        result['dist_from_median'] = dist_from_median

    return result


if __name__ == '__main__':
    # Test on a few trees
    DATA_DIR = Path(r'E:\tree_id_2.0\data')
    IMAGE_BASE = Path(r'D:\Task')

    print("Loading data...")
    df_outliers = pd.read_excel(DATA_DIR / 'photos_by_tree_with_outliers.xlsx')

    # Find trees with many photos for testing
    tree_counts = df_outliers.groupby(['address', 'tree_number']).size().reset_index(name='count')
    tree_counts = tree_counts.sort_values('count', ascending=False)

    print(f"\nTesting triangulation on trees with most photos...")
    print("="*60)

    results = []
    for _, row in tree_counts.head(10).iterrows():
        result = test_triangulation_on_tree(
            row['address'],
            row['tree_number'],
            df_outliers,
            IMAGE_BASE
        )
        if result:
            result['address'] = row['address']
            result['tree_number'] = row['tree_number']
            results.append(result)
        print("="*60)

    # Summary
    print("\n\nSUMMARY")
    print("="*60)
    triangulated = [r for r in results if r.get('method') == 'triangulation']
    print(f"Successfully triangulated: {len(triangulated)}/{len(results)} trees")

    if triangulated:
        dists = [r['dist_from_median'] for r in triangulated]
        print(f"Distance from median - Mean: {np.mean(dists):.1f}m, Max: {np.max(dists):.1f}m")

        high_conf = [r for r in triangulated if r['confidence'] == 'high']
        print(f"High confidence results: {len(high_conf)}")
