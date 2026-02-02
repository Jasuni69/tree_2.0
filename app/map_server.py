"""
Flask web app - Interactive map with address pins.
Run: python map_server.py
Open: http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, request
from pathlib import Path
import json
import statistics
import re

app = Flask(__name__)

DATA_DIR = Path(r'E:\tree_id_2.0\data')

# Type categories based on TreeNumber prefix
TYPE_CATEGORIES = {
    'tree': ['Träd', 'Tree', 'Solitär', 'Tr d'],
    'bush': ['Buskyta', 'Buske', 'Busk'],
    'grass': ['Gräsyta', 'Gräs', 'Ängsyta'],
    'planting': ['Planteringsyta', 'Plantering', 'Planting'],
    'other': []  # Catch-all for unmatched
}

def categorize_tree_number(tree_number: str) -> str:
    """Categorize a tree number into type category."""
    tn = str(tree_number).lower()

    for category, prefixes in TYPE_CATEGORIES.items():
        for prefix in prefixes:
            if prefix.lower() in tn:
                return category

    return 'other'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tree Address Map</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; display: flex; height: 100vh; }

        #sidebar {
            width: 350px;
            background: #f5f5f5;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #ddd;
        }

        #map { flex: 1; height: 100%; }

        h1 { font-size: 1.4em; margin-bottom: 10px; color: #333; }
        h2 { font-size: 1.1em; margin: 15px 0 10px; color: #555; }

        .stats {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .stat-box {
            background: #fff;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            flex: 1;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #2196F3; }
        .stat-label { font-size: 0.8em; color: #666; }

        .filter-section {
            background: #fff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .filter-section h3 { margin: 0 0 10px; font-size: 1em; color: #333; }
        .filter-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
            cursor: pointer;
        }
        .filter-item input { margin-right: 10px; cursor: pointer; }
        .filter-item label { cursor: pointer; flex: 1; }
        .filter-count { color: #666; font-size: 0.85em; }

        .color-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .color-tree { background: #2E7D32; }
        .color-bush { background: #7B1FA2; }
        .color-grass { background: #FFD600; }
        .color-planting { background: #E65100; }
        .color-other { background: #616161; }

        #address-list {
            max-height: 300px;
            overflow-y: auto;
            background: #fff;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .address-item {
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            transition: background 0.2s;
        }
        .address-item:hover { background: #e3f2fd; }
        .address-item.selected { background: #bbdefb; }
        .address-name { font-weight: bold; color: #333; }
        .address-meta { font-size: 0.85em; color: #666; margin-top: 3px; }

        #details {
            margin-top: 20px;
            background: #fff;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: none;
        }
        #details.visible { display: block; }

        .tree-list { margin-top: 10px; max-height: 150px; overflow-y: auto; }
        .tree-item { padding: 5px 10px; background: #f9f9f9; margin: 3px 0; border-radius: 3px; }

        .refresh-btn {
            margin-top: 15px;
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        .refresh-btn:hover { background: #1976D2; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Tree Address Map</h1>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value" id="addr-count">-</div>
                <div class="stat-label">Addresses</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="tree-count">-</div>
                <div class="stat-label">Items</div>
            </div>
        </div>

        <div class="filter-section">
            <h3>Filter by Type</h3>
            <div class="filter-item">
                <input type="checkbox" id="filter-tree" checked onchange="applyFilters()">
                <span class="color-dot color-tree"></span>
                <label for="filter-tree">Träd (Trees)</label>
                <span class="filter-count" id="count-tree">0</span>
            </div>
            <div class="filter-item">
                <input type="checkbox" id="filter-bush" checked onchange="applyFilters()">
                <span class="color-dot color-bush"></span>
                <label for="filter-bush">Buskyta (Bushes)</label>
                <span class="filter-count" id="count-bush">0</span>
            </div>
            <div class="filter-item">
                <input type="checkbox" id="filter-grass" checked onchange="applyFilters()">
                <span class="color-dot color-grass"></span>
                <label for="filter-grass">Gräsyta (Grass)</label>
                <span class="filter-count" id="count-grass">0</span>
            </div>
            <div class="filter-item">
                <input type="checkbox" id="filter-planting" checked onchange="applyFilters()">
                <span class="color-dot color-planting"></span>
                <label for="filter-planting">Planteringsyta</label>
                <span class="filter-count" id="count-planting">0</span>
            </div>
            <div class="filter-item">
                <input type="checkbox" id="filter-other" checked onchange="applyFilters()">
                <span class="color-dot color-other"></span>
                <label for="filter-other">Other</label>
                <span class="filter-count" id="count-other">0</span>
            </div>
        </div>

        <h2>Addresses</h2>
        <div id="address-list"></div>

        <div id="details">
            <h2 id="detail-title">Address Details</h2>
            <p><strong>Coordinates:</strong> <span id="detail-coords"></span></p>
            <p><strong>Trees:</strong> <span id="detail-tree-count"></span></p>
            <p><strong>GPS Samples:</strong> <span id="detail-samples"></span></p>
            <div class="tree-list" id="tree-list"></div>
        </div>

        <button class="refresh-btn" onclick="loadData()">Refresh Data</button>
    </div>

    <div id="map"></div>

    <script>
        // Initialize map centered on Stockholm area
        const map = L.map('map').setView([59.33, 18.07], 11);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        let markers = [];
        let addressData = [];
        let selectedAddress = null;
        let initialLoadDone = false;

        const typeColors = {
            'tree': '#2E7D32',      // Dark green
            'bush': '#7B1FA2',      // Purple
            'grass': '#FFD600',     // Yellow
            'planting': '#E65100',  // Deep orange
            'other': '#616161'      // Gray
        };

        function loadData() {
            fetch('/api/addresses')
                .then(r => r.json())
                .then(data => {
                    addressData = data;
                    updateTypeCounts();
                    applyFilters();
                })
                .catch(err => console.error('Error loading data:', err));
        }

        function updateTypeCounts() {
            const counts = { tree: 0, bush: 0, grass: 0, planting: 0, other: 0 };
            addressData.forEach(addr => {
                addr.trees_by_type = addr.trees_by_type || {};
                Object.keys(counts).forEach(type => {
                    counts[type] += addr.trees_by_type[type] || 0;
                });
            });
            Object.keys(counts).forEach(type => {
                document.getElementById('count-' + type).textContent = counts[type];
            });
        }

        function getActiveFilters() {
            return {
                tree: document.getElementById('filter-tree').checked,
                bush: document.getElementById('filter-bush').checked,
                grass: document.getElementById('filter-grass').checked,
                planting: document.getElementById('filter-planting').checked,
                other: document.getElementById('filter-other').checked
            };
        }

        function applyFilters() {
            const filters = getActiveFilters();

            // Filter addresses that have at least one visible type
            const filteredData = addressData.filter(addr => {
                const typeData = addr.trees_by_type || {};
                return Object.keys(filters).some(type => filters[type] && (typeData[type] || 0) > 0);
            });

            updateUI(filteredData, filters);
        }

        function updateUI(filteredData, filters) {
            // Update stats
            document.getElementById('addr-count').textContent = filteredData.length;
            const totalItems = filteredData.reduce((sum, a) => {
                const typeData = a.trees_by_type || {};
                return sum + Object.keys(filters).reduce((s, type) => {
                    return s + (filters[type] ? (typeData[type] || 0) : 0);
                }, 0);
            }, 0);
            document.getElementById('tree-count').textContent = totalItems;

            // Clear existing markers
            markers.forEach(m => map.removeLayer(m.marker));
            markers = [];

            // Add markers and build address list
            const listEl = document.getElementById('address-list');
            listEl.innerHTML = '';

            // Sort by item count
            filteredData.sort((a, b) => b.tree_count - a.tree_count);

            filteredData.forEach(addr => {
                // Determine dominant type for marker color
                const typeData = addr.trees_by_type || {};
                let dominantType = 'other';
                let maxCount = 0;
                Object.keys(typeData).forEach(type => {
                    if (filters[type] && typeData[type] > maxCount) {
                        maxCount = typeData[type];
                        dominantType = type;
                    }
                });

                const color = typeColors[dominantType] || '#2196F3';

                // Add marker
                const marker = L.circleMarker([addr.lat, addr.lon], {
                    radius: 8,
                    fillColor: color,
                    color: color,
                    weight: 2,
                    fillOpacity: 0.7
                }).addTo(map);

                // Calculate filtered count for popup
                const popupCount = Object.keys(filters).reduce((sum, type) => {
                    return sum + (filters[type] ? (typeData[type] || 0) : 0);
                }, 0);
                marker.bindPopup(`<b>${addr.address}</b><br>Items: ${popupCount}`);
                marker.on('click', () => selectAddress(addr.address));
                markers.push({ marker, address: addr.address, type: dominantType });

                // Calculate filtered item count for this address
                const filteredCount = Object.keys(filters).reduce((sum, type) => {
                    return sum + (filters[type] ? (typeData[type] || 0) : 0);
                }, 0);

                // Add to list
                const item = document.createElement('div');
                item.className = 'address-item';
                item.dataset.address = addr.address;
                item.innerHTML = `
                    <div class="address-name">${addr.address}</div>
                    <div class="address-meta">${filteredCount} items · ${addr.sample_count} samples</div>
                `;
                item.onclick = () => selectAddress(addr.address);
                listEl.appendChild(item);
            });

            // Fit map to markers only on initial load
            if (markers.length > 0 && !initialLoadDone) {
                const group = L.featureGroup(markers.map(m => m.marker));
                map.fitBounds(group.getBounds().pad(0.1));
                initialLoadDone = true;
            }
        }

        function selectAddress(addrName) {
            selectedAddress = addrName;
            const addr = addressData.find(a => a.address === addrName);
            if (!addr) return;

            // Update list selection
            document.querySelectorAll('.address-item').forEach(el => {
                el.classList.toggle('selected', el.dataset.address === addrName);
            });

            // Update markers
            markers.forEach(m => {
                const isSelected = m.address === addrName;
                m.marker.setStyle({
                    fillColor: isSelected ? '#f44336' : (typeColors[m.type] || '#2196F3'),
                    color: isSelected ? '#c62828' : (typeColors[m.type] || '#1565C0'),
                    radius: isSelected ? 12 : 8
                });
                if (isSelected) m.marker.bringToFront();
            });

            // Zoom to address
            map.setView([addr.lat, addr.lon], 16);

            // Show details
            document.getElementById('details').classList.add('visible');
            document.getElementById('detail-title').textContent = addr.address;
            document.getElementById('detail-coords').textContent = `${addr.lat.toFixed(6)}, ${addr.lon.toFixed(6)}`;
            document.getElementById('detail-tree-count').textContent = addr.tree_count;
            document.getElementById('detail-samples').textContent = addr.sample_count;

            // Tree list
            const treeListEl = document.getElementById('tree-list');
            treeListEl.innerHTML = '';
            addr.trees.sort().forEach(tree => {
                const item = document.createElement('div');
                item.className = 'tree-item';
                item.textContent = tree;
                treeListEl.appendChild(item);
            });
        }

        // Load data on page load
        loadData();
    </script>
</body>
</html>
"""

def load_address_data():
    """Load and aggregate GPS data by address."""
    results_path = DATA_DIR / 'gps_extraction_results.json'
    if not results_path.exists():
        return []

    with open(results_path) as f:
        gps_results = json.load(f)

    # Aggregate by address
    address_data = {}
    for r in gps_results:
        if not r.get('gps'):
            continue

        addr = r['address']
        if addr not in address_data:
            address_data[addr] = {
                'lats': [],
                'lons': [],
                'trees': set(),
                'trees_by_type': {'tree': set(), 'bush': set(), 'grass': set(), 'planting': set(), 'other': set()},
                'task_count': 0
            }

        address_data[addr]['lats'].append(r['gps']['lat'])
        address_data[addr]['lons'].append(r['gps']['lon'])
        address_data[addr]['trees'].add(r['tree_number'])
        address_data[addr]['task_count'] += 1

        # Categorize tree
        category = categorize_tree_number(r['tree_number'])
        address_data[addr]['trees_by_type'][category].add(r['tree_number'])

    # Build result with median coords
    result = []
    for addr, data in address_data.items():
        if len(data['lats']) > 0:
            trees_by_type = {k: len(v) for k, v in data['trees_by_type'].items()}
            result.append({
                'address': addr,
                'lat': statistics.median(data['lats']),
                'lon': statistics.median(data['lons']),
                'tree_count': len(data['trees']),
                'sample_count': len(data['lats']),
                'trees': sorted(list(data['trees'])),
                'trees_by_type': trees_by_type
            })

    return result


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/addresses')
def get_addresses():
    return jsonify(load_address_data())


if __name__ == '__main__':
    print("Starting Tree Address Map server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
