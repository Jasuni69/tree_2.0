"""
Diagnostic Map Tool - View location clusters and outlier photos.
Run: python diagnostic_map.py
Open: http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, send_file, abort, request
from pathlib import Path
import json
import math
import re
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def normalize_tree_number(tree_str):
    """Extract just the number from tree names like 'Träd 1', 'Tree1', 'träd 1'."""
    if pd.isna(tree_str):
        return None
    # Find all numbers in the string
    numbers = re.findall(r'\d+', str(tree_str))
    if numbers:
        return int(numbers[0])
    return None


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def cluster_photos_by_radius(df, radius_meters):
    """Cluster photos using DBSCAN with haversine distance."""
    coords = np.radians(df[['photo_lat', 'photo_lon']].values)
    # DBSCAN with haversine metric, eps in radians (radius / earth_radius)
    eps_radians = radius_meters / 6371000
    db = DBSCAN(eps=eps_radians, min_samples=1, metric='haversine')
    df = df.copy()
    df['cluster_id'] = db.fit_predict(coords)
    return df

app = Flask(__name__)

DATA_DIR = Path(r'E:\tree_id_2.0\data')
IMAGE_BASE = Path(r'D:\Task')

def normalize_address(s):
    """Normalize address string - strip and collapse whitespace."""
    if pd.isna(s):
        return ''
    return ' '.join(str(s).strip().split())

# Load data at startup
print("Loading data...")
df_clusters = pd.read_excel(DATA_DIR / 'photos_by_location_cluster.xlsx')
df_outliers = pd.read_excel(DATA_DIR / 'photos_by_tree_with_outliers.xlsx')

# Add normalized address column for lookups
df_outliers['address_norm'] = df_outliers['address'].apply(normalize_address)

print(f"Loaded {len(df_clusters)} clustered photos, {len(df_outliers)} outlier data rows")

# Load Tibram ground truth
def load_tibram_ground_truth():
    """Load Tibram ground truth coordinates."""
    tibram_path = DATA_DIR.parent / 'Tibram Trees.xlsx'
    if not tibram_path.exists():
        print("Tibram Trees.xlsx not found - ground truth disabled")
        return None

    tibram = pd.read_excel(tibram_path, skiprows=3, header=None)
    tibram = tibram.dropna(axis=1, how='all')
    tibram.columns = ['Location', 'TreeNumber', 'TreeCategory', 'Age', 'Latitude', 'Longitude', 'Polygon', 'DeadStatus']
    tibram = tibram[~tibram['Location'].isin(['Location'])]

    # Normalize location
    tibram['loc_norm'] = tibram['Location'].apply(lambda s: ' '.join(str(s).strip().split()) if pd.notna(s) else '')
    tibram['tree_num'] = tibram['TreeNumber'].apply(normalize_tree_number)

    # Filter to trees with coordinates
    tibram = tibram[(tibram['Latitude'].notna()) & (tibram['TreeCategory'].isin(['Tree', 'Grey Tree']))]
    tibram['lat'] = pd.to_numeric(tibram['Latitude'], errors='coerce')
    tibram['lon'] = pd.to_numeric(tibram['Longitude'], errors='coerce')

    # Create lookup key - ensure int for tree number
    tibram['tree_num_int'] = tibram['tree_num'].apply(lambda x: int(x) if pd.notna(x) else -1)
    tibram['key'] = tibram['loc_norm'] + '|' + tibram['tree_num_int'].astype(str)

    print(f"Loaded {len(tibram)} Tibram ground truth positions")
    return tibram

df_tibram = load_tibram_ground_truth()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tree Data Diagnostic Tool</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; display: flex; height: 100vh; }

        #sidebar {
            width: 400px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #ddd;
        }

        #controls {
            padding: 15px;
            background: #fff;
            border-bottom: 1px solid #ddd;
        }

        h1 { font-size: 1.2em; margin-bottom: 10px; color: #333; }
        h2 { font-size: 1em; margin: 10px 0; color: #555; }

        .mode-toggle {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .mode-btn {
            flex: 1;
            padding: 10px;
            border: 2px solid #ddd;
            background: #fff;
            cursor: pointer;
            font-size: 0.9em;
            border-radius: 5px;
        }
        .mode-btn.active {
            border-color: #2196F3;
            background: #e3f2fd;
        }

        .filter-row {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            align-items: center;
        }
        .filter-row label { font-size: 0.85em; }
        .filter-row input, .filter-row select {
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }

        #info-panel {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }

        .cluster-info, .photo-info {
            background: #fff;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .cluster-info h3 { font-size: 0.95em; color: #333; margin-bottom: 5px; }
        .cluster-info p { font-size: 0.85em; color: #666; margin: 3px 0; }
        .tree-list { font-size: 0.8em; color: #888; margin-top: 5px; max-height: 60px; overflow-y: auto; }

        .photo-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .photo-card {
            background: #fff;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            cursor: pointer;
        }
        .photo-card img {
            width: 100%;
            height: 120px;
            object-fit: cover;
            background: #eee;
        }
        .photo-card .meta {
            padding: 8px;
            font-size: 0.75em;
        }
        .photo-card .tree-num { font-weight: bold; color: #333; }
        .photo-card .distance { color: #e53935; }
        .photo-card .distance.ok { color: #43a047; }

        #map { flex: 1; height: 100%; }

        #image-modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        #image-modal.visible { display: flex; }
        #image-modal img {
            max-width: 90%;
            max-height: 80%;
            object-fit: contain;
        }
        #image-modal .caption {
            color: #fff;
            margin-top: 15px;
            text-align: center;
            font-size: 0.9em;
        }
        #image-modal .close-btn {
            position: absolute;
            top: 20px;
            right: 30px;
            color: #fff;
            font-size: 2em;
            cursor: pointer;
        }

        .legend {
            background: #fff;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.8em;
            margin-top: 10px;
        }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <div id="controls">
            <h1>Diagnostic Tool</h1>

            <div class="mode-toggle">
                <button class="mode-btn active" data-mode="clusters" onclick="setMode('clusters')">
                    Clusters
                </button>
                <button class="mode-btn" data-mode="outliers" onclick="setMode('outliers')">
                    Outliers
                </button>
                <button class="mode-btn" data-mode="spread" onclick="setMode('spread')">
                    Tree Spread
                </button>
            </div>

            <div id="cluster-filters">
                <div class="filter-row">
                    <label>Cluster radius (m):</label>
                    <input type="range" id="cluster-radius" value="15" min="5" max="50" step="5" oninput="updateRadiusLabel()" onchange="loadData()">
                    <span id="radius-label">15m</span>
                </div>
                <div class="filter-row">
                    <label>Min trees at location:</label>
                    <input type="number" id="min-trees" value="2" min="1" max="50" onchange="applyFilters()">
                </div>
            </div>

            <div id="outlier-filters" style="display:none;">
                <div class="filter-row">
                    <label>Min distance (m):</label>
                    <input type="number" id="min-distance" value="30" min="0" max="1000" onchange="applyFilters()">
                </div>
            </div>

            <div id="spread-filters" style="display:none;">
                <div class="filter-row">
                    <label>Address:</label>
                    <select id="address-select" onchange="loadTreesForAddress()" style="flex:1; max-width:200px;">
                        <option value="">-- Select Address --</option>
                    </select>
                </div>
                <div class="filter-row">
                    <label>Tree:</label>
                    <select id="tree-select" onchange="showTreeSpread()" style="flex:1; max-width:200px;">
                        <option value="">-- Select Tree --</option>
                    </select>
                </div>
                <div class="filter-row">
                    <label>OK threshold (m):</label>
                    <input type="range" id="spread-threshold" value="10" min="1" max="20" step="1" oninput="updateThresholdLabel()" onchange="updateSpreadColors()">
                    <span id="threshold-label">10m</span>
                </div>
                <div id="spread-info" style="font-size:0.85em; color:#666; margin-top:5px;"></div>
                <div id="spread-stats" style="font-size:0.85em; margin-top:5px;"></div>
            </div>

            <div class="filter-row">
                <label>Year:</label>
                <select id="year-filter" onchange="loadData()">
                    <option value="all">All Years</option>
                    <option value="2023">2023</option>
                    <option value="2024">2024</option>
                    <option value="2025">2025</option>
                </select>
            </div>

            <div class="legend">
                <div class="legend-item"><span class="legend-dot" style="background:#000000"></span> Ground Truth (Tibram)</div>
                <div class="legend-item"><span class="legend-dot" style="background:#2196F3"></span> Median (from photos)</div>
                <div class="legend-item"><span class="legend-dot" style="background:#43a047"></span> Photo within threshold</div>
                <div class="legend-item"><span class="legend-dot" style="background:#e53935"></span> Photo outside threshold</div>
            </div>
        </div>

        <div id="info-panel">
            <p style="color:#666; font-size:0.9em;">Click a marker on the map to view details and photos.</p>
        </div>
    </div>

    <div id="map"></div>

    <div id="image-modal">
        <span class="close-btn" onclick="closeModal()">&times;</span>
        <img id="modal-img" src="" alt="">
        <div class="caption" id="modal-caption"></div>
    </div>

    <script>
        const map = L.map('map').setView([59.33, 18.07], 11);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap'
        }).addTo(map);

        let markers = [];
        let currentMode = 'clusters';
        let clusterData = [];
        let outlierData = [];

        function updateRadiusLabel() {
            const radius = document.getElementById('cluster-radius').value;
            document.getElementById('radius-label').textContent = radius + 'm';
        }

        function updateThresholdLabel() {
            const threshold = document.getElementById('spread-threshold').value;
            document.getElementById('threshold-label').textContent = threshold + 'm';
        }

        function updateSpreadColors() {
            if (!treeSpreadData || !treeSpreadData.photos) return;

            const threshold = parseFloat(document.getElementById('spread-threshold').value);
            const photos = treeSpreadData.photos;

            // Count within/outside threshold
            const withinCount = photos.filter(p => p.distance <= threshold).length;
            const outsideCount = photos.length - withinCount;

            document.getElementById('spread-stats').innerHTML =
                `<span style="color:#43a047">● ${withinCount} within ${threshold}m</span> | ` +
                `<span style="color:#e53935">● ${outsideCount} outside</span>`;

            // Update marker colors (skip first marker which is median)
            markers.slice(1).forEach((marker, i) => {
                const photo = photos[i];
                if (photo) {
                    const color = photo.distance > threshold ? '#e53935' : '#43a047';
                    marker.setStyle({ fillColor: color, color: color });
                }
            });

            // Update photo cards
            document.querySelectorAll('.photo-card .distance').forEach((el, i) => {
                const photo = photos[i];
                if (photo) {
                    el.className = photo.distance > threshold ? 'distance' : 'distance ok';
                }
            });
        }

        // Load data
        function loadData() {
            const year = document.getElementById('year-filter').value;
            const radius = document.getElementById('cluster-radius').value;
            document.getElementById('info-panel').innerHTML = '<p style="color:#666;">Loading clusters with ' + radius + 'm radius...</p>';

            fetch(`/api/clusters/dynamic?year=${year}&radius=${radius}`).then(r => r.json()).then(data => {
                clusterData = data;
                applyFilters();
            });
            fetch(`/api/outliers?year=${year}`).then(r => r.json()).then(data => {
                outlierData = data;
                if (currentMode === 'outliers') applyFilters();
            });
        }
        loadData();

        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
            document.getElementById('cluster-filters').style.display = mode === 'clusters' ? 'block' : 'none';
            document.getElementById('outlier-filters').style.display = mode === 'outliers' ? 'block' : 'none';
            document.getElementById('spread-filters').style.display = mode === 'spread' ? 'block' : 'none';
            document.getElementById('info-panel').innerHTML = '<p style="color:#666; font-size:0.9em;">Click a marker on the map to view details.</p>';

            if (mode === 'spread') {
                loadAddresses();
                markers.forEach(m => map.removeLayer(m));
                markers = [];
            } else {
                applyFilters();
            }
        }

        let addressData = [];
        let treeSpreadData = [];

        function loadAddresses() {
            fetch('/api/addresses').then(r => r.json()).then(data => {
                addressData = data;
                const select = document.getElementById('address-select');
                select.innerHTML = '<option value="">-- Select Address (' + data.length + ') --</option>';
                data.forEach(addr => {
                    select.innerHTML += `<option value="${addr.address}">${addr.address} (${addr.tree_count} trees)</option>`;
                });
            });
        }

        function loadTreesForAddress() {
            const address = document.getElementById('address-select').value;
            const treeSelect = document.getElementById('tree-select');
            treeSelect.innerHTML = '<option value="">-- Select Tree --</option>';
            document.getElementById('spread-info').innerHTML = '';

            if (!address) return;

            fetch(`/api/address/${encodeURIComponent(address)}/trees`).then(r => r.json()).then(data => {
                data.forEach(tree => {
                    const spreadInfo = tree.spread > 20 ? ' ⚠️' : '';
                    const variantInfo = tree.variants && tree.variants.length > 0 ? ' *' : '';
                    treeSelect.innerHTML += `<option value="${tree.tree_number}">Tree ${tree.tree_number} (${tree.photo_count} photos, ${tree.spread.toFixed(0)}m)${spreadInfo}${variantInfo}</option>`;
                });
                if (data.some(t => t.variants && t.variants.length > 0)) {
                    document.getElementById('spread-info').innerHTML = '* = multiple naming variants merged';
                }
            });
        }

        function showTreeSpread() {
            const address = document.getElementById('address-select').value;
            const treeNumber = document.getElementById('tree-select').value;

            if (!address || !treeNumber) return;

            markers.forEach(m => map.removeLayer(m));
            markers = [];

            fetch(`/api/tree/${encodeURIComponent(address)}/${encodeURIComponent(treeNumber)}/spread`)
                .then(r => r.json())
                .then(data => {
                    treeSpreadData = data;
                    const threshold = parseFloat(document.getElementById('spread-threshold').value);

                    // Show info
                    let infoHtml = `<strong>Spread: ${data.spread.toFixed(1)}m</strong><br>` +
                        `Photos: ${data.photos.length}<br>` +
                        `Median: ${data.median_lat.toFixed(5)}, ${data.median_lon.toFixed(5)}`;

                    if (data.ground_truth) {
                        infoHtml += `<br><strong style="color:#000">Ground Truth: ${data.ground_truth.lat.toFixed(5)}, ${data.ground_truth.lon.toFixed(5)}</strong>`;
                        infoHtml += `<br><span style="color:#666">Median-GT distance: ${data.ground_truth.distance_from_median.toFixed(1)}m</span>`;
                    } else {
                        infoHtml += `<br><span style="color:#999">No ground truth available</span>`;
                    }
                    document.getElementById('spread-info').innerHTML = infoHtml;

                    // Add ground truth marker (black) if available
                    if (data.ground_truth) {
                        const gtMarker = L.circleMarker([data.ground_truth.lat, data.ground_truth.lon], {
                            radius: 14,
                            fillColor: '#000000',
                            color: '#000000',
                            weight: 3,
                            fillOpacity: 0.9
                        }).addTo(map);
                        gtMarker.bindTooltip(`Ground Truth (${data.ground_truth.distance_from_median.toFixed(1)}m from median)`, {permanent: false});
                        markers.push(gtMarker);
                    }

                    // Add median marker (blue)
                    const medianMarker = L.circleMarker([data.median_lat, data.median_lon], {
                        radius: 12,
                        fillColor: '#2196F3',
                        color: '#1565C0',
                        weight: 3,
                        fillOpacity: 0.8
                    }).addTo(map);
                    medianMarker.bindTooltip('Median position (from photos)', {permanent: false});
                    markers.push(medianMarker);

                    // Add photo markers
                    data.photos.forEach((p, i) => {
                        const color = p.distance > threshold ? '#e53935' : '#43a047';

                        const marker = L.circleMarker([p.lat, p.lon], {
                            radius: 8,
                            fillColor: color,
                            color: color,
                            weight: 2,
                            fillOpacity: 0.7
                        }).addTo(map);

                        marker.bindTooltip(`${p.distance.toFixed(1)}m from median`, {permanent: false});
                        marker.on('click', () => showSpreadPhoto(p, address, treeNumber));
                        markers.push(marker);
                    });

                    // Fit map to markers
                    if (markers.length > 0) {
                        const group = L.featureGroup(markers);
                        map.fitBounds(group.getBounds().pad(0.2));
                    }

                    // Show photos in panel
                    showSpreadPhotos(data.photos, address, treeNumber);

                    // Update stats
                    updateSpreadColors();
                });
        }

        function showSpreadPhoto(photo, address, treeNumber) {
            showImage(photo.image_path, treeNumber, address + ' - ' + photo.distance.toFixed(1) + 'm from median');
        }

        function showSpreadPhotos(photos, address, treeNumber) {
            const threshold = parseFloat(document.getElementById('spread-threshold').value);
            let html = `<h2>Photos for Tree ${treeNumber}</h2><div class="photo-grid">`;
            photos.forEach(p => {
                const distClass = p.distance > threshold ? 'distance' : 'distance ok';
                html += `
                    <div class="photo-card" onclick="showImage('${p.image_path}', '${treeNumber}', '${address}')">
                        <img src="/image/${encodeURIComponent(p.image_path)}" alt="${treeNumber}"
                             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><text x=%2210%22 y=%2250%22>No image</text></svg>'">
                        <div class="meta">
                            <div class="tree-num">Task: ${p.task_id}</div>
                            <div class="${distClass}">${p.distance.toFixed(1)}m from median</div>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            document.getElementById('info-panel').innerHTML = html;
        }

        function applyFilters() {
            markers.forEach(m => map.removeLayer(m));
            markers = [];

            if (currentMode === 'clusters') {
                const minTrees = parseInt(document.getElementById('min-trees').value) || 2;
                const filtered = clusterData.filter(c => c.unique_trees >= minTrees);

                filtered.forEach(cluster => {
                    const color = cluster.unique_trees > 10 ? '#e53935' :
                                  cluster.unique_trees > 3 ? '#ff9800' : '#43a047';

                    const marker = L.circleMarker([cluster.lat, cluster.lon], {
                        radius: Math.min(6 + cluster.unique_trees, 20),
                        fillColor: color,
                        color: color,
                        weight: 2,
                        fillOpacity: 0.7
                    }).addTo(map);

                    marker.on('click', () => showClusterDetails(cluster));
                    markers.push(marker);
                });

                if (markers.length > 0 && !map._loaded) {
                    const group = L.featureGroup(markers);
                    map.fitBounds(group.getBounds().pad(0.1));
                    map._loaded = true;
                }
            } else {
                // Outliers mode - group by tree
                const minDist = parseInt(document.getElementById('min-distance').value) || 30;
                const outliers = outlierData.filter(o => o.distance > minDist);

                // Group by tree and show worst outlier location
                const byTree = {};
                outliers.forEach(o => {
                    const key = o.address + '|' + o.tree_number;
                    if (!byTree[key] || o.distance > byTree[key].distance) {
                        byTree[key] = o;
                    }
                });

                Object.values(byTree).forEach(o => {
                    const color = o.distance > 100 ? '#e53935' :
                                  o.distance > 50 ? '#ff9800' : '#43a047';

                    const marker = L.circleMarker([o.photo_lat, o.photo_lon], {
                        radius: 8,
                        fillColor: color,
                        color: color,
                        weight: 2,
                        fillOpacity: 0.7
                    }).addTo(map);

                    marker.on('click', () => showOutlierDetails(o.address, o.tree_number));
                    markers.push(marker);
                });
            }
        }

        function showClusterDetails(cluster) {
            const year = document.getElementById('year-filter').value;
            const radius = document.getElementById('cluster-radius').value;
            fetch(`/api/cluster/dynamic/${cluster.id}/photos?year=${year}&radius=${radius}`)
                .then(r => r.json())
                .then(photos => {
                    let html = `
                        <div class="cluster-info">
                            <h3>Cluster #${cluster.id}</h3>
                            <p><strong>Location:</strong> ${cluster.lat.toFixed(5)}, ${cluster.lon.toFixed(5)}</p>
                            <p><strong>Photos:</strong> ${cluster.photo_count}</p>
                            <p><strong>Different trees:</strong> ${cluster.unique_trees}</p>
                            <p><strong>Address:</strong> ${cluster.address}</p>
                            <div class="tree-list"><strong>Trees:</strong> ${cluster.trees}</div>
                        </div>
                        <h2>Photos at this location (${photos.length})</h2>
                        <div class="photo-grid">
                    `;

                    photos.slice(0, 20).forEach(p => {
                        html += `
                            <div class="photo-card" onclick="showImage('${p.image_path}', '${p.tree_number}', '${p.address}')">
                                <img src="/image/${encodeURIComponent(p.image_path)}" alt="${p.tree_number}"
                                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><text x=%2210%22 y=%2250%22>No image</text></svg>'">
                                <div class="meta">
                                    <div class="tree-num">${p.tree_number}</div>
                                    <div>Task: ${p.task_id}</div>
                                </div>
                            </div>
                        `;
                    });

                    html += '</div>';
                    if (photos.length > 20) {
                        html += `<p style="margin-top:10px;color:#666;font-size:0.85em;">Showing 20 of ${photos.length} photos</p>`;
                    }

                    document.getElementById('info-panel').innerHTML = html;
                });
        }

        function showOutlierDetails(address, treeNumber) {
            fetch(`/api/tree/${encodeURIComponent(address)}/${encodeURIComponent(treeNumber)}/photos`)
                .then(r => r.json())
                .then(data => {
                    let html = `
                        <div class="cluster-info">
                            <h3>${treeNumber}</h3>
                            <p><strong>Address:</strong> ${address}</p>
                            <p><strong>Tree median:</strong> ${data.median_lat.toFixed(5)}, ${data.median_lon.toFixed(5)}</p>
                            <p><strong>Total photos:</strong> ${data.photos.length}</p>
                        </div>
                        <h2>Photos for this tree</h2>
                        <div class="photo-grid">
                    `;

                    data.photos.forEach(p => {
                        const distClass = p.distance > 30 ? 'distance' : 'distance ok';
                        html += `
                            <div class="photo-card" onclick="showImage('${p.image_path}', '${treeNumber}', '${address}')">
                                <img src="/image/${encodeURIComponent(p.image_path)}" alt="${treeNumber}"
                                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><text x=%2210%22 y=%2250%22>No image</text></svg>'">
                                <div class="meta">
                                    <div class="tree-num">Task: ${p.task_id}</div>
                                    <div class="${distClass}">${p.distance.toFixed(1)}m from median</div>
                                </div>
                            </div>
                        `;
                    });

                    html += '</div>';
                    document.getElementById('info-panel').innerHTML = html;
                });
        }

        function showImage(path, tree, address) {
            document.getElementById('modal-img').src = '/image/' + encodeURIComponent(path);
            document.getElementById('modal-caption').innerHTML = `<strong>${tree}</strong><br>${address}<br>${path}`;
            document.getElementById('image-modal').classList.add('visible');
        }

        function closeModal() {
            document.getElementById('image-modal').classList.remove('visible');
        }

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') closeModal();
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/clusters')
def get_clusters():
    """Get unique clusters with summary info (pre-computed 15m radius)."""
    year = request.args.get('year', 'all')

    df = df_clusters
    if year != 'all':
        df = df[df['image_path'].str[:4] == year]

    if len(df) == 0:
        return jsonify([])

    clusters = df.groupby('cluster_id').agg({
        'cluster_lat': 'first',
        'cluster_lon': 'first',
        'photo_lat': 'count',
        'tree_number': lambda x: len(x.unique()),
        'trees_in_cluster': 'first',
        'addresses_in_cluster': 'first'
    }).reset_index()

    result = []
    for _, row in clusters.iterrows():
        result.append({
            'id': int(row['cluster_id']),
            'lat': row['cluster_lat'],
            'lon': row['cluster_lon'],
            'photo_count': int(row['photo_lat']),
            'unique_trees': int(row['tree_number']),
            'trees': row['trees_in_cluster'][:200] if pd.notna(row['trees_in_cluster']) else '',
            'address': row['addresses_in_cluster'][:100] if pd.notna(row['addresses_in_cluster']) else ''
        })
    return jsonify(result)


@app.route('/api/clusters/dynamic')
def get_clusters_dynamic():
    """Cluster photos dynamically with adjustable radius."""
    year = request.args.get('year', 'all')
    radius = float(request.args.get('radius', 15))

    # Use outliers data which has raw photo coordinates
    df = df_outliers[['task_id', 'address', 'tree_number', 'image_path', 'photo_lat', 'photo_lon']].copy()

    if year != 'all':
        df = df[df['image_path'].str[:4] == year]

    if len(df) == 0:
        return jsonify([])

    # Cluster using DBSCAN (fast)
    df = cluster_photos_by_radius(df, radius)

    # Aggregate clusters
    result = []
    for cid in df['cluster_id'].unique():
        cluster_df = df[df['cluster_id'] == cid]
        lat = cluster_df['photo_lat'].mean()
        lon = cluster_df['photo_lon'].mean()
        photo_count = len(cluster_df)
        unique_trees = cluster_df['tree_number'].nunique()
        trees = ', '.join(sorted(cluster_df['tree_number'].unique())[:20])
        addresses = ', '.join(sorted(cluster_df['address'].unique())[:5])

        result.append({
            'id': int(cid),
            'lat': lat,
            'lon': lon,
            'photo_count': photo_count,
            'unique_trees': unique_trees,
            'trees': trees[:200],
            'address': addresses[:100]
        })

    return jsonify(result)


@app.route('/api/cluster/dynamic/<int:cluster_id>/photos')
def get_dynamic_cluster_photos(cluster_id):
    """Get photos for a dynamically created cluster."""
    year = request.args.get('year', 'all')
    radius = float(request.args.get('radius', 15))

    df = df_outliers[['task_id', 'address', 'tree_number', 'image_path', 'photo_lat', 'photo_lon']].copy()

    if year != 'all':
        df = df[df['image_path'].str[:4] == year]

    if len(df) == 0:
        return jsonify([])

    # Cluster using DBSCAN (fast)
    df = cluster_photos_by_radius(df, radius)
    cluster_photos = df[df['cluster_id'] == cluster_id]

    result = []
    for _, row in cluster_photos.iterrows():
        result.append({
            'task_id': str(row['task_id']),
            'tree_number': row['tree_number'],
            'address': row['address'],
            'image_path': row['image_path'],
            'lat': row['photo_lat'],
            'lon': row['photo_lon']
        })
    return jsonify(result)


@app.route('/api/cluster/<int:cluster_id>/photos')
def get_cluster_photos(cluster_id):
    """Get all photos in a cluster."""
    cluster_photos = df_clusters[df_clusters['cluster_id'] == cluster_id]
    result = []
    for _, row in cluster_photos.iterrows():
        result.append({
            'task_id': str(row['task_id']),
            'tree_number': row['tree_number'],
            'address': row['address'],
            'image_path': row['image_path'],
            'lat': row['photo_lat'],
            'lon': row['photo_lon']
        })
    return jsonify(result)


@app.route('/api/outliers')
def get_outliers():
    """Get outlier summary data."""
    year = request.args.get('year', 'all')

    df = df_outliers
    if year != 'all':
        df = df[df['image_path'].str[:4] == year]

    result = []
    for _, row in df.iterrows():
        result.append({
            'address': row['address'],
            'tree_number': row['tree_number'],
            'task_id': str(row['task_id']),
            'image_path': row['image_path'],
            'photo_lat': row['photo_lat'],
            'photo_lon': row['photo_lon'],
            'distance': row['distance_from_tree_median_m']
        })
    return jsonify(result)


@app.route('/api/tree/<path:address>/<tree_number>/photos')
def get_tree_photos(address, tree_number):
    """Get all photos for a specific tree with distances."""
    tree_photos = df_outliers[
        (df_outliers['address'] == address) &
        (df_outliers['tree_number'] == tree_number)
    ].sort_values('distance_from_tree_median_m', ascending=False)

    if len(tree_photos) == 0:
        return jsonify({'photos': [], 'median_lat': 0, 'median_lon': 0})

    median_lat = tree_photos['tree_median_lat'].iloc[0]
    median_lon = tree_photos['tree_median_lon'].iloc[0]

    photos = []
    for _, row in tree_photos.iterrows():
        photos.append({
            'task_id': str(row['task_id']),
            'image_path': row['image_path'],
            'lat': row['photo_lat'],
            'lon': row['photo_lon'],
            'distance': row['distance_from_tree_median_m']
        })

    return jsonify({
        'photos': photos,
        'median_lat': median_lat,
        'median_lon': median_lon
    })


@app.route('/api/addresses')
def get_addresses():
    """Get list of unique addresses with tree counts."""
    address_stats = df_outliers.groupby('address').agg({
        'tree_number': lambda x: len(x.unique()),
        'task_id': 'count'
    }).reset_index()
    address_stats.columns = ['address', 'tree_count', 'photo_count']
    address_stats = address_stats.sort_values('address')

    result = []
    for _, row in address_stats.iterrows():
        result.append({
            'address': row['address'],
            'tree_count': int(row['tree_count']),
            'photo_count': int(row['photo_count'])
        })
    return jsonify(result)


@app.route('/api/address/<path:address>/trees')
def get_trees_for_address(address):
    """Get trees at an address with spread info, grouped by normalized number."""
    addr_data = df_outliers[df_outliers['address'] == address].copy()

    if len(addr_data) == 0:
        return jsonify([])

    # Add normalized tree number
    addr_data['tree_num_normalized'] = addr_data['tree_number'].apply(normalize_tree_number)

    result = []
    for norm_num in sorted(addr_data['tree_num_normalized'].dropna().unique()):
        tree_data = addr_data[addr_data['tree_num_normalized'] == norm_num]
        photo_count = len(tree_data)
        # Use first original tree_number as display name
        display_name = tree_data['tree_number'].iloc[0]
        # Show all variants if different
        variants = tree_data['tree_number'].unique()

        # Calculate spread
        if photo_count > 1:
            coords = tree_data[['photo_lat', 'photo_lon']].values
            max_dist = 0
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    max_dist = max(max_dist, dist)
            spread = max_dist
        else:
            spread = 0

        result.append({
            'tree_number': str(int(norm_num)),  # Use normalized number
            'display_name': display_name,
            'variants': list(variants) if len(variants) > 1 else [],
            'photo_count': photo_count,
            'spread': spread
        })

    return jsonify(result)


@app.route('/api/tree/<path:address>/<tree_number>/spread')
def get_tree_spread(address, tree_number):
    """Get detailed spread data for a specific tree (uses normalized number)."""
    # Normalize address for lookup
    addr_norm = normalize_address(address)
    addr_data = df_outliers[df_outliers['address_norm'] == addr_norm].copy()
    addr_data['tree_num_normalized'] = addr_data['tree_number'].apply(normalize_tree_number)

    # Match by normalized number
    try:
        norm_num = int(tree_number)
    except:
        norm_num = normalize_tree_number(tree_number)

    tree_data = addr_data[addr_data['tree_num_normalized'] == norm_num]

    if len(tree_data) == 0:
        return jsonify({'photos': [], 'median_lat': 0, 'median_lon': 0, 'spread': 0})

    # Calculate median from all matching photos
    median_lat = tree_data['photo_lat'].median()
    median_lon = tree_data['photo_lon'].median()

    # Calculate spread
    coords = tree_data[['photo_lat', 'photo_lon']].values
    max_dist = 0
    if len(coords) > 1:
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                max_dist = max(max_dist, dist)

    photos = []
    for _, row in tree_data.iterrows():
        # Calculate distance from median
        dist_from_median = haversine_distance(row['photo_lat'], row['photo_lon'], median_lat, median_lon)
        photos.append({
            'task_id': str(row['task_id']),
            'image_path': row['image_path'],
            'lat': row['photo_lat'],
            'lon': row['photo_lon'],
            'distance': dist_from_median,
            'original_tree': row['tree_number']
        })

    # Sort by distance descending
    photos.sort(key=lambda x: -x['distance'])

    # Look up ground truth from Tibram
    ground_truth = None
    if df_tibram is not None:
        # Normalize address for lookup
        addr_norm = ' '.join(str(address).strip().split())
        lookup_key = f"{addr_norm}|{norm_num}"
        tibram_match = df_tibram[df_tibram['key'] == lookup_key]
        if len(tibram_match) > 0:
            gt_lat = tibram_match['lat'].iloc[0]
            gt_lon = tibram_match['lon'].iloc[0]
            gt_dist_from_median = haversine_distance(median_lat, median_lon, gt_lat, gt_lon)
            ground_truth = {
                'lat': gt_lat,
                'lon': gt_lon,
                'distance_from_median': gt_dist_from_median
            }

    return jsonify({
        'photos': photos,
        'median_lat': median_lat,
        'median_lon': median_lon,
        'spread': max_dist,
        'ground_truth': ground_truth
    })


@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve image from D: drive."""
    full_path = IMAGE_BASE / image_path
    if full_path.exists():
        return send_file(full_path, mimetype='image/png')
    else:
        abort(404)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Diagnostic Map Tool")
    print("="*50)
    print("Open http://localhost:5000 in your browser")
    print("\nModes:")
    print("  - Location Clusters: See photos grouped by GPS location")
    print("  - Outliers by Tree: See photos far from their tree median")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
