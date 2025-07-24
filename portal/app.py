import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json

st.set_page_config(
    page_title="FAF Food Flows Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* Global Styles */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  * {
    font-family: 'Inter', sans-serif;
  }

  /* Hide default elements */
  #MainMenu, footer, header {visibility:hidden;}
  [data-testid="collapsedControl"] {display: none !important;}

  /* Main container styling */
  .block-container {
    padding: 0 1rem;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    min-height: 100vh;
  }

  /* Sidebar styling */
  .css-1d391kg {
    background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    border-right: 1px solid #475569;
  }

  .css-1d391kg .element-container {
    margin-bottom: 1.5rem;
  }

  /* Button styling */
  button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    border-radius: 12px;
    color: white;
    font-weight: 600;
    border: none;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
  }

  button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
  }

  /* Heading styles */
  h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    text-align: center;
    letter-spacing: -0.02em;
  }

  h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #374151;
    margin-top: 0;
    margin-bottom: 1rem;
  }

  h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #4b5563;
    margin-bottom: 0.75rem;
  }

  /* Welcome screen styling */
  .welcome-container {
    text-align: center;
    padding: 5rem 3rem;
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 24px;
    margin: 2rem 0;
    box-shadow: 0 20px 60px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
  }

  .welcome-title {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.5rem;
    text-shadow: none;
    letter-spacing: -0.03em;
  }

  .welcome-subtitle {
    font-size: 1.6rem;
    color: #6b7280;
    margin-bottom: 3rem;
    line-height: 1.7;
    font-weight: 400;
  }

  .start-button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    border: none;
    padding: 1.25rem 3.5rem;
    font-size: 1.4rem;
    font-weight: 700;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.4s ease;
    box-shadow: 0 8px 30px rgba(5, 150, 105, 0.3);
    letter-spacing: 0.02em;
  }

  .start-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(5, 150, 105, 0.4);
  }

  /* Feature list styling */
  .feature-list {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 3rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.06);
  }

  .feature-list h3 {
    color: #1f2937;
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .feature-list ul {
    list-style: none;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
  }

  .feature-list li {
    background: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    border: 1px solid #e5e7eb;
    transition: all 0.3s ease;
  }

  .feature-list li:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
  }

  /* Card styling */
  .metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid rgba(255,255,255,0.8);
    margin-bottom: 1rem;
  }

  /* Expander styling */
  .stExpanderHeader {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
    color: #0c4a6e !important;
    border-radius: 12px !important;
    border: 1px solid #bae6fd !important;
    font-weight: 600 !important;
  }

  /* Table styling */
  .stTable {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  }

  .stTable th {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    color: #1f2937 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #e5e7eb !important;
  }

  .stTable td {
    background: white !important;
    color: #374151 !important;
    border-bottom: 1px solid #f3f4f6 !important;
  }

  .stTable tr:hover td {
    background: #f8fafc !important;
  }

  /* Metric styling */
  [data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
  }

  [data-testid="stMetricValue"] {
    color: #059669 !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
  }

  /* Selectbox styling */
  .stSelectbox > div > div {
    border-radius: 8px !important;
    border: 2px solid #e5e7eb !important;
  }

  .stSelectbox > div > div:hover {
    border-color: #3b82f6 !important;
  }

  /* Slider styling */
  .stSlider > div > div > div > div {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
  }

  .stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    color: #6b7280;
    font-weight: 500;
  }

  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    border-color: #3b82f6;
  }

  /* Download button styling */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 8px;
    color: white;
    font-weight: 600;
    border: none;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
  }

  .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
  }

  /* Warning styling */
  .stAlert {
    border-radius: 12px;
    border: 1px solid #fef3c7;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
  }

  /* Code block styling */
  code {
    background: #f1f5f9;
    color: #0f172a;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.875rem;
    border: 1px solid #e2e8f0;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .welcome-title {
      font-size: 2.5rem;
    }

    .welcome-subtitle {
      font-size: 1.2rem;
    }

    .feature-list ul {
      grid-template-columns: 1fr;
    }
  }
</style>
""", unsafe_allow_html=True)

# Initialize session state for app state
if 'app_started' not in st.session_state:
    st.session_state['app_started'] = False

FAF_FILES = {
    "01 - Live Animals/Fish": "predicted_sctg_1.csv",
    "02 - Cereal Grains": "predicted_sctg_2.csv",
    "03 - Other Ag products.": "predicted_sctg_3.csv",
    "04 - Animal Feed": "predicted_sctg_4.csv",
    "05 - Meat/Seafood": "predicted_sctg_5.csv",
    "06 - Milled Grain Prods.": "predicted_sctg_6.csv",
    "07 - Other Foodstuffs": "predicted_sctg_7.csv"
}

@st.cache_data
def load_sctg_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(f"./cleaned_data/{file_name}")
    df['origin'] = df['origin'].astype(str).str.zfill(5)
    df['dest']   = df['dest'].astype(str).str.zfill(5)
    df = df.query("exist_prob > 0.5 and predicted_value_original > 0")
    return df

@st.cache_data
def load_county_metadata() -> pd.DataFrame:
    meta = pd.read_csv("cleaned_data/state_and_county_fips_master.csv")
    meta['fips'] = meta['fips'].astype(str).str.zfill(5)
    meta = meta.rename(columns={"fips": "FIPS", "name": "County", "state": "State"})
    return meta[['FIPS', 'County', 'State']]

@st.cache_data
def load_county_boundaries() -> dict:
    gdf = gpd.read_file("data/shapefiles/cb_2017_us_county_500k/cb_2017_us_county_500k.shp")
    gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)
    gdf = gdf.rename(columns={'GEOID': 'FIPS'}).merge(load_county_metadata(), on="FIPS", how="left")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
    return json.loads(gdf.to_json())

def convert_sctg_to_trip(sctg_df, meta_df):
    # Create origin and destination labels with county names
    origin_labels = []
    dest_labels = []

    for orig, dest in zip(sctg_df["origin"], sctg_df["dest"]):
        # Get origin county info
        orig_info = meta_df[meta_df.FIPS == orig]
        if len(orig_info) > 0:
            orig_label = f"{orig_info['County'].iloc[0]}, {orig_info['State'].iloc[0]}"
        else:
            orig_label = f"Unknown ({orig})"

        # Get destination county info
        dest_info = meta_df[meta_df.FIPS == dest]
        if len(dest_info) > 0:
            dest_label = f"{dest_info['County'].iloc[0]}, {dest_info['State'].iloc[0]}"
        else:
            dest_label = f"Unknown ({dest})"

        origin_labels.append(orig_label)
        dest_labels.append(dest_label)

    return pd.DataFrame({
        "coordinates": [
            [[ox, oy], [dx, dy]]
            for ox, oy, dx, dy in zip(
                sctg_df["origin_x"], sctg_df["origin_y"],
                sctg_df["dest_x"], sctg_df["dest_y"]
            )
        ],
        "timestamps": [[0, 2]] * len(sctg_df),
        "orig_dms": origin_labels,
        "dest_dms": dest_labels,
        "origin": sctg_df["origin"].tolist(),
        "dest": sctg_df["dest"].tolist(),
        "predicted_value_original": sctg_df["predicted_value_original"].tolist()
    })

# Load data
META  = load_county_metadata()
BNDJS = load_county_boundaries()

# Welcome Screen or Main App
if not st.session_state['app_started']:
    # Welcome Screen
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">🌾 Food Flow Portal</div>
        <div class="welcome-subtitle">
            Welcome to the FAF Food Flows Dashboard!<br>
            Explore food transportation patterns across the United States using advanced Graph Neural Network predictions.<br>
            Discover how food products flow between counties and states in 2017.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Exploring", key="start_button", use_container_width=True):
            st.session_state['app_started'] = True
            st.rerun()

    st.markdown("""
    <div class="feature-list">
        <h3>What you can explore:</h3>
        <ul>
            <li>Interactive map with food flow arcs</li>
            <li>Filter by food categories and counties</li>
            <li>View top origins and destinations</li>
            <li>Analyze transportation patterns</li>
            <li>Download the data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    # Main App Content
    # Sidebar
    with st.sidebar:
        st.image("./image/image17.png")

        # Back to Welcome button
        if st.button("Back to Welcome", key="back_button"):
            st.session_state['app_started'] = False
            st.rerun()

        st.markdown("---")

        st.markdown("## Filter Options")
        if 'origin_lbl' not in st.session_state:
            st.session_state['origin_lbl'] = None
        if 'dest_lbl' not in st.session_state:
            st.session_state['dest_lbl'] = "All"

        cat = st.selectbox("Food Category (SCTG)", list(FAF_FILES))
        flows = load_sctg_data(FAF_FILES[cat])

        origins = sorted(flows['origin'].unique())
        origin_lbls = [f"{f} - {META.loc[META.FIPS==f,'County'].iat[0]} - {META.loc[META.FIPS==f,'State'].iat[0]}"
                       if f in META.FIPS.values else f for f in origins]
        origin_lbls = ["All"] + origin_lbls
        default_idx = origin_lbls.index("06037 - Los Angeles County - CA") if "06037 - Los Angeles County - CA" in origin_lbls else 0

        if st.session_state['origin_lbl'] is None:
            st.session_state['origin_lbl'] = origin_lbls[default_idx]
        origin_lbl = st.selectbox("Origin County", origin_lbls, index=origin_lbls.index(st.session_state['origin_lbl']) if st.session_state['origin_lbl'] in origin_lbls else default_idx, key="origin_select")
        st.session_state['origin_lbl'] = origin_lbl
        origin_fip = origin_lbl.split(" - ")[0] if origin_lbl != "All" else "All"

        if origin_lbl == "All":
            dests = sorted(flows['dest'].unique())
            filt = flows.copy()
        else:
            dests = sorted(flows.query("origin == @origin_fip")['dest'].unique())
            filt = flows.query("origin == @origin_fip")

        dest_lbls = [f"{d} - {META.loc[META.FIPS==d,'County'].iat[0]} - {META.loc[META.FIPS==d,'State'].iat[0]}"
                     if d in META.FIPS.values else d for d in dests]
        dest_options = ["All"] + dest_lbls
        if st.session_state['dest_lbl'] not in dest_options:
            st.session_state['dest_lbl'] = "All"
        dest_lbl = st.selectbox("Destination County", dest_options, index=dest_options.index(st.session_state['dest_lbl']), key="dest_select")
        st.session_state['dest_lbl'] = dest_lbl
        if dest_lbl != "All":
            dest_fip = dest_lbl.split(" - ")[0]
            filt = filt.query("dest == @dest_fip")

        if st.button("\u21c6 Swap Origin and Destination"):
            if st.session_state['dest_lbl'] == "All":
                st.session_state['origin_lbl'] = "All"
                st.session_state['dest_lbl'] = origin_lbl
            else:
                st.session_state['origin_lbl'], st.session_state['dest_lbl'] = st.session_state['dest_lbl'], st.session_state['origin_lbl']
            st.rerun()

        st.markdown("## Trip Summary")

        # Create metric cards with better styling
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #3b82f6; margin-bottom: 0.5rem;">
                        {len(filt):,}
                    </div>
                    <div style="font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                        Number of Trips
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #059669; margin-bottom: 0.5rem;">
                        {filt['predicted_value_original'].sum():,.0f}
                    </div>
                    <div style="font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                        Total Tons Shipped
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        top_n = st.slider("Top links to render", min_value=1, max_value=300, value=60, step=1)

    # Main panel
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🌾 GNN Food Flow Portal</h1>
        <p style="color: #6b7280; font-size: 1.1rem; margin-top: -0.5rem;">
            Advanced Graph Neural Network Food Transportation Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["How to Use", "Interactive Map", "Download"])

    with tab1:
        st.markdown("## How to Use the Food Flow Portal")

        st.markdown("""
        ### Welcome to the FAF Food Flows Dashboard!

        This interactive dashboard allows you to explore food transportation patterns across the United States using advanced Graph Neural Network predictions.

        ### Getting Started

        1. **Select Food Category**: Choose from 7 different food categories in the sidebar
           - Live Animals/Fish
           - Cereal Grains
           - Other Agricultural Products
           - Animal Feed
           - Meat/Seafood
           - Milled Grain Products
           - Other Foodstuffs

        2. **Filter by Location**:
           - **Origin County**: Select where food shipments originate from
           - **Destination County**: Select where food shipments are going to
           - Use "All" to see all origins or destinations

        3. **Explore the Map**:
           - Interactive 3D map shows food flow arcs between counties
           - Hover over arcs to see origin and destination details
           - Adjust the number of top links to display (25-200)

        4. **Analyze Data**:
           - View trip summary metrics in the sidebar
           - See top origins or destinations in the expandable table
           - Download filtered data for further analysis

        ### Interactive Features

        - **Swap Origin/Destination**: Quickly reverse your analysis
        - **Zoom & Pan**: Navigate the map to focus on specific regions
        - **Hover Effects**: Get detailed information about each flow
        - **Filter Combinations**: Mix and match categories and locations

        ### Understanding the Data

        - **Predicted Value**: Tons of food shipped (based on GNN predictions)
        - **Existence Probability**: Confidence level of the predicted flow
        - **FIPS Codes**: Federal Information Processing Standards county identifiers

        ### Tips for Best Experience

        - Start with "All" origins to see major food flow patterns
        - Focus on specific counties for detailed local analysis
        - Use different food categories to understand supply chains
        - Combine filters to isolate specific trade relationships
        """)

        st.markdown("---")
        st.markdown("### Ready to Explore?")
        st.markdown("Switch to the **Interactive Map** tab to start analyzing food flows!")

    with tab2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    padding: 1.5rem;
                    border-radius: 16px;
                    border: 1px solid #bae6fd;
                    margin-bottom: 1.5rem;">
            <p style="margin: 0; color: #0369a1; font-size: 1.1rem; font-weight: 500;">
                Trips from <strong>{origin_lbl}</strong> to <strong>{dest_lbl}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        top_links = filt.nlargest(top_n, 'predicted_value_original')
        arc_df = convert_sctg_to_trip(top_links, META)

        boundary_layer = pdk.Layer(
            "GeoJsonLayer", BNDJS,
            stroked=True, filled=True,
            get_fill_color=[248, 250, 252, 40],
            get_line_color=[203, 213, 225, 100],
            line_width_min_pixels=0.5,
            pickable=False,
        )

        arc_layer = pdk.Layer(
            "ArcLayer", arc_df,
            get_source_position="coordinates[0]",
            get_target_position="coordinates[1]",
            get_width="predicted_value_original",
            get_tilt=15,
            get_source_color=[5, 150, 105, 180],
            get_target_color=[239, 68, 68, 180],
            get_width_scale=0.0001,
            get_width_min_pixels=1,
            get_width_max_pixels=8,
            pickable=True,
            auto_highlight=True,
        )

        view = pdk.ViewState(latitude=39.8283, longitude=-98.5795, zoom=3, pitch=45)

                        # Add click functionality
        if 'clicked_flow' not in st.session_state:
            st.session_state['clicked_flow'] = None





        st.pydeck_chart(
            pdk.Deck(
                layers=[boundary_layer, arc_layer],
                initial_view_state=view,
                tooltip={
                    "html": """
                    <div style="background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
                                color: white;
                                padding: 12px;
                                border-radius: 8px;
                                border: 1px solid #4b5563;
                                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                                font-family: 'Inter', sans-serif;">
                        <div style="margin-bottom: 8px;">
                            <span style="color: #10b981; font-weight: 600;">Origin:</span> {orig_dms}
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="color: #ef4444; font-weight: 600;">Destination:</span> {dest_dms}
                        </div>
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #4b5563;">
                            <span style="color: #3b82f6; font-weight: 600;">FIPS: {origin} → {dest}</span>
                        </div>
                    </div>
                    """,
                    "style": {"backgroundColor": "transparent", "border": "none"}
                }
            ),
            use_container_width=True,
            height=480,
        )


        # col1, col2 = st.columns([1, 3])

        # with col1:
        #     if st.button("🗺️ Reset Map View", help="Reset the map to the default view"):
        #         # Reset map view to initial state
        #         st.session_state['map_view'] = {
        #             'latitude': 39.8283,
        #             'longitude': -98.5795,
        #             'zoom': 3,
        #             'pitch': 45,
        #             'bearing': 0
        #         }

        # with col2:
        #     st.markdown("💡 **Tip:** Use the mouse to pan and zoom the map. Double-click to reset the view.")
        origin = st.session_state['origin_lbl']
        dest = st.session_state['dest_lbl']
        if origin == "All" and dest != "All":
            expander_title = "Top 5 Origins by Tons"
            group_col = 'origin'
        else:
            expander_title = "Top 5 Destinations by Tons"
            group_col = 'dest'

        with st.expander(expander_title, expanded=True):
            table = (filt.groupby(group_col)['predicted_value_original']
                          .sum()
                          .sort_values(ascending=False)
                          .head(5)
                          .reset_index())

            table['County'] = table[group_col].map(lambda f: META.loc[META.FIPS==f, 'County'].iat[0] if f in META.FIPS.values else "")
            table['State'] = table[group_col].map(lambda f: META.loc[META.FIPS==f, 'State'].iat[0] if f in META.FIPS.values else "")

            st.table(table.rename(columns={group_col: 'FIPS', 'predicted_value_original': 'Tons Shipped'}))

    with tab3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                    padding: 1.5rem;
                    border-radius: 16px;
                    border: 1px solid #bbf7d0;
                    margin-bottom: 2rem;">
            <h2 style="margin: 0 0 0.5rem 0; color: #166534;">Download Data</h2>
            <p style="margin: 0; color: #15803d; font-size: 1.1rem;">
                Export your filtered data in various formats for further analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📥 Download Options")

        # Create download buttons for different data formats
        col1, col3 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #e5e7eb;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
                        height: 100%;">
                <h4 style="margin: 0 0 0.75rem 0; color: #1f2937;">Current Filtered Data</h4>
                <p style="margin: 0 0 1.5rem 0; color: #6b7280; font-size: 0.9rem;">
                    Download the data currently displayed on the map based on your selected filters.
                </p>
            """, unsafe_allow_html=True)

            # Add vertical space before the download button
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            if len(filt) > 0:
                csv_data = filt.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"food_flows_{cat.replace(' ', '_').lower()}_{origin_lbl.split(' - ')[0] if origin_lbl != 'All' else 'all'}_{dest_lbl.split(' - ')[0] if dest_lbl != 'All' else 'all'}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data available for current filters")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #e5e7eb;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
                        height: 100%;">
                <h4 style="margin: 0 0 0.75rem 0; color: #1f2937;">Summary Statistics</h4>
                <p style="margin: 0 0 1.5rem 0; color: #6b7280; font-size: 0.9rem;">
                    Download summary statistics for the current food category.
                </p>
            """, unsafe_allow_html=True)

            # Add vertical space before the download button
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            if len(filt) > 0:
                summary_stats = filt.groupby(['origin', 'dest'])['predicted_value_original'].sum().reset_index()
                summary_stats = summary_stats.merge(META, left_on='origin', right_on='FIPS', how='left')
                summary_stats = summary_stats.merge(META, left_on='dest', right_on='FIPS', how='left', suffixes=('_origin', '_dest'))
                summary_stats = summary_stats[['origin', 'County_origin', 'State_origin', 'dest', 'County_dest', 'State_dest', 'predicted_value_original']]
                summary_stats.columns = ['Origin_FIPS', 'Origin_County', 'Origin_State', 'Dest_FIPS', 'Dest_County', 'Dest_State', 'Tons_Shipped']

                summary_csv = summary_stats.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=summary_csv,
                    file_name=f"food_flows_summary_{cat.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data available for summary")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        # --- Data Description Card ---
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #fde68a;
                        margin-bottom: 1.5rem;">
                <h3 style="margin: 0 0 1rem 0; color: #92400e;">Data Description</h3>
                <div style="margin-bottom: 1.5rem;">
                    <h4 style="color: #92400e; margin-bottom: 0.5rem;">Food Flow Data Fields:</h4>
                    <table style="color: #78350f; font-size: 0.98rem; border-collapse: collapse;">
                        <tr><td style="padding: 2px 8px;"><code>origin</code></td><td>Origin county FIPS code</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>dest</code></td><td>Destination county FIPS code</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>origin_x</code>, <code>origin_y</code></td><td>Origin county coordinates</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>dest_x</code>, <code>dest_y</code></td><td>Destination county coordinates</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>predicted_value_original</code></td><td>Tons of food shipped (GNN prediction)</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>exist_prob</code></td><td>Probability that this flow exists</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>sctg</code></td><td>Food category code (1-7)</td></tr>
                    </table>
                </div>
                <div style="margin-bottom: 1.5rem;">
                    <h4 style="color: #92400e; margin-bottom: 0.5rem;">County Metadata Fields:</h4>
                    <table style="color: #78350f; font-size: 0.98rem; border-collapse: collapse;">
                        <tr><td style="padding: 2px 8px;"><code>FIPS</code></td><td>Federal Information Processing Standards county code</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>County</code></td><td>County name</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>State</code></td><td>State name</td></tr>
                    </table>
                </div>
                <p style="color: #78350f; margin: 0; font-weight: 500;">
                    <strong>Data Source:</strong> FAF (Freight Analysis Framework) 2017 with Graph Neural Network predictions
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Filter Information Card ---
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #bae6fd;">
                <h3 style="margin: 0 0 1rem 0; color: #0c4a6e;">Filter Information</h3>
                <div style="color: #0369a1; line-height: 1.6;">
                    <p style="margin: 0 0 0.5rem 0;"><strong>Current Filters Applied:</strong></p>
                    <table style="margin: 0; padding-left: 0; color: #0369a1; font-size: 1rem;">
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Food Category:</strong></td>
                            <td>{cat}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Origin:</strong></td>
                            <td>{origin_lbl}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Destination:</strong></td>
                            <td>{dest_lbl}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Total Records:</strong></td>
                            <td>{len(filt):,}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Total Tons:</strong></td>
                            <td>{filt['predicted_value_original'].sum():,.1f}</td>
                        </tr>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
