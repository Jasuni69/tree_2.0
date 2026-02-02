# Data Quality Report - Tree Identification Project

**Date:** 2026-01-21
**Data Source:** Tasks 2023-2025.xlsx + D:\Task images

---

## Overview

| Metric | Count |
|--------|-------|
| Task records in Excel | 136,953 |
| Images found on D: drive | 58,113 (42%) |
| Images with GPS coordinates | 23,479 |
| Tree photos (excl. bushes, grass, etc.) | 17,819 |
| Unique tree locations | 2,330 |
| Unique addresses | 345 |

---

## Problem 1: GPS Spread per Tree

Photos for the same tree should cluster within ~10m (photographer stands 0-5m away, GPS accuracy ±3-5m).

| Confidence | Spread | Trees | % | Status |
|------------|--------|-------|---|--------|
| HIGH | ≤10m | 413 | 22% | ✓ Good |
| MEDIUM | 10-20m | 441 | 23% | ⚠️ Acceptable |
| LOW | >20m | 1,047 | 55% | ❌ Problematic |

**55% of trees have LOW confidence** - photos claiming to be the same tree are spread over 20+ meters.

### Worst Examples

| Address | Tree | Spread |
|---------|------|--------|
| Rissneleden | Träd 27 | 526m |
| Skärholmens Moské | Träd 1 | 1,675m |
| Lindhagensgatan 49 | Träd 2 | 6,781km (!!) |

---

## Problem 2: Outlier Photos

Photos that are far from their own tree's calculated center.

| Distance from tree median | Photos | % |
|---------------------------|--------|---|
| 0-5m | 9,053 | 51% |
| 5-10m | 3,980 | 22% |
| 10-15m | 1,788 | 10% |
| 15-20m | 996 | 6% |
| 20-30m | 951 | 5% |
| >30m | 1,051 | 6% |

**1,051 photos** are >30m from their tree's median location - likely wrong tree selected during data entry.

---

## Problem 3: Multiple Trees at Same GPS Location

When clustering photos by actual GPS coordinates (15m radius), we found:

- **2,054 location clusters** total
- **1,008 clusters** have multiple different tree numbers at the same spot

### Examples

| Address | Different trees at same spot | Photos |
|---------|------------------------------|--------|
| Eksätravägen Skärholmsvägen | 28 trees | 42 |
| Tyska Botten Väg 9 | 19 trees | 242 |
| Sankt Göransparken E2 | 17 trees | 418 |
| Torsplan etapp 2 | 16 trees | 64 |
| Gustav Adolfs Torg | 12 trees | 63 |

**Possible causes:**
1. Trees are physically very close together (<15m spacing)
2. Workers photograph from same standing position for multiple trees
3. Data entry errors (wrong tree selected in app)

---

## Possible Causes of Data Issues

1. **Worker standing too far from tree** - Expected 0-5m, actual may be 10-20m+

2. **Wrong tree selected in app** - Easy to mis-tap on dropdown

3. **GPS drift/error** - Some devices have poor GPS accuracy

4. **Vague addresses** - "Rissneleden" spans 500m+, multiple trees with same number?

5. **No verification step** - Data goes directly to database without review

---

## Questions to Investigate

1. **How do workers select tree numbers?**
   - Dropdown list? Manual entry? Map selection?
   - Is it easy to accidentally select wrong tree?

2. **What is actual tree spacing at problem sites?**
   - Are trees at Tyska Botten actually within 15m of each other?
   - Or is 19 trees at same spot definitely an error?

3. **Is there independent ground truth?**
   - Survey data with actual tree coordinates?
   - Site maps or planting records?

4. **Can we access the worker app UI?**
   - Understand how errors happen
   - Suggest improvements

---

## Files for Review

All located in `E:\tree_id_2.0\data\`

| File | Description |
|------|-------------|
| `photos_by_tree_with_outliers.xlsx` | Each photo with distance from its tree's median |
| `photos_by_location_cluster.xlsx` | Photos grouped by GPS location (15m clusters) |
| `outlier_photos.xlsx` | Only photos >30m from their tree median |
| `trees_with_coordinates.csv` | All tree locations with confidence scores |
| `tree_locations.json` | Tree centroids + spread data (JSON format) |

---

## Recommendations

### Short term
1. Review the outlier photos manually
2. Visit problem sites (Tyska Botten, Sankt Göransparken) to verify tree spacing
3. Identify patterns in worker errors

### Long term
1. Add GPS validation in worker app (warn if photo far from expected tree location)
2. Add visual verification step (CNN comparison to previous photos of same tree)
3. Flag suspicious entries for supervisor review before saving

---

## Conclusion

Current data quality is concerning:
- **55% of trees** have unreliable location data (>20m spread)
- **1,051 photos** are clearly assigned to wrong tree (>30m off)
- **1,008 locations** have multiple conflicting tree assignments

**Before using this as ground truth for ML training, data cleanup is needed.**
