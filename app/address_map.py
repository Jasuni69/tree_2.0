"""
Interactive map showing all addresses with tree data.
Each pin = one address. Click to see trees at that address.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import json
import statistics

# Config
DATA_DIR = Path(r'E:\tree_id_2.0\data')
EXCEL_PATH = Path(r'E:\tree_id_new\data\excel_files\Tasks 2023-2025.xlsx')

st.set_page_config(
    page_title="Tree Address Map",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_gps_results():
    """Load GPS extraction results."""
    results_path = DATA_DIR / 'gps_extraction_results.json'
    if not results_path.exists():
        return []
    with open(results_path) as f:
        return json.load(f)

@st.cache_data
def load_excel_data():
    """Load full Excel data for tree info."""
    return pd.read_excel(EXCEL_PATH)

@st.cache_data
def build_address_coords(gps_results):
    """Aggregate GPS coords per address using median."""
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
                'task_count': 0
            }

        address_data[addr]['lats'].append(r['gps']['lat'])
        address_data[addr]['lons'].append(r['gps']['lon'])
        address_data[addr]['trees'].add(r['tree_number'])
        address_data[addr]['task_count'] += 1

    # Calculate median coords
    result = []
    for addr, data in address_data.items():
        if len(data['lats']) > 0:
            result.append({
                'address': addr,
                'lat': statistics.median(data['lats']),
                'lon': statistics.median(data['lons']),
                'tree_count': len(data['trees']),
                'sample_count': len(data['lats']),
                'trees': list(data['trees'])
            })

    return result

def main():
    st.title("Tree Address Map")

    # Load data
    gps_results = load_gps_results()

    if not gps_results:
        st.warning("No GPS data yet. Run extract_gps.py first.")
        st.info("Extraction may still be in progress. Refresh to see updates.")
        return

    # Build address coordinates
    address_coords = build_address_coords(gps_results)

    if not address_coords:
        st.warning("No addresses with GPS coordinates found yet.")
        return

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Addresses with GPS", len(address_coords))
    with col2:
        total_trees = sum(a['tree_count'] for a in address_coords)
        st.metric("Trees Mapped", total_trees)
    with col3:
        total_samples = sum(a['sample_count'] for a in address_coords)
        st.metric("GPS Samples", total_samples)

    # Sidebar - Address list
    st.sidebar.header("Addresses")
    address_names = sorted([a['address'] for a in address_coords])
    selected_address = st.sidebar.selectbox(
        "Select Address",
        ["All Addresses"] + address_names
    )

    # Create map
    if selected_address == "All Addresses":
        # Center on all addresses
        center_lat = statistics.mean([a['lat'] for a in address_coords])
        center_lon = statistics.mean([a['lon'] for a in address_coords])
        zoom = 11
    else:
        # Center on selected address
        addr_data = next(a for a in address_coords if a['address'] == selected_address)
        center_lat = addr_data['lat']
        center_lon = addr_data['lon']
        zoom = 16

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    # Add markers
    for addr in address_coords:
        is_selected = (selected_address != "All Addresses" and addr['address'] == selected_address)

        color = 'red' if is_selected else 'blue'
        radius = 12 if is_selected else 8

        popup_html = f"""
        <b>{addr['address']}</b><br>
        Trees: {addr['tree_count']}<br>
        GPS samples: {addr['sample_count']}
        """

        folium.CircleMarker(
            location=[addr['lat'], addr['lon']],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=addr['address']
        ).add_to(m)

    # Display map
    st.subheader("Map")
    map_data = st_folium(m, width=1000, height=500)

    # Show details for selected address
    if selected_address != "All Addresses":
        addr_data = next(a for a in address_coords if a['address'] == selected_address)

        st.subheader(f"Details: {selected_address}")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Coordinates:** {addr_data['lat']:.6f}, {addr_data['lon']:.6f}")
            st.write(f"**Number of trees:** {addr_data['tree_count']}")
            st.write(f"**GPS samples:** {addr_data['sample_count']}")

        with col2:
            st.write("**Trees at this address:**")
            for tree in sorted(addr_data['trees']):
                st.write(f"  - {tree}")

    # Table view
    st.subheader("All Addresses")
    df = pd.DataFrame(address_coords)
    df = df[['address', 'lat', 'lon', 'tree_count', 'sample_count']]
    df = df.sort_values('tree_count', ascending=False)
    st.dataframe(df, use_container_width=True)

    # Refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == '__main__':
    main()
