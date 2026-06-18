# Section 1 — Page Config & Imports
"""
app/app.py
-----------
Streamlit web application for House Price Prediction.

Run with:
    streamlit run app/app.py
"""

import sys
import os

# Allow imports from src/ when running from app/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

from src.predict import HousePricePredictor

# Must be the FIRST Streamlit command
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Section 2 — Load Model (Cached)
@st.cache_resource
def load_predictor():
    """
    Load the prediction pipeline once and cache it.
    Streamlit reruns the whole script on every interaction —
    without caching, this would reload the model every time.
    """
    try:
        return HousePricePredictor()
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found!\n\n{str(e)}")
        st.stop()


predictor = load_predictor()

# Section 3 — Header & Description
# ── HEADER ──────────────────────────────────────────────────────────
st.title("🏠 California House Price Predictor")
st.markdown(
    """
    Predict median house prices in California using a machine learning model
    trained on real housing data. Adjust the inputs in the sidebar and click
    **Predict** to see the estimated price.
    """
)
st.divider()

# Section 4 — Sidebar Inputs
# ── SIDEBAR: USER INPUTS ───────────────────────────────────────────
st.sidebar.header("🏘️ House Features")
st.sidebar.markdown("Adjust the values below to describe the house.")

# Income
med_inc = st.sidebar.slider(
    "Median Income (in $10,000s)",
    min_value=0.5, max_value=15.0, value=5.0, step=0.1,
    help="Median household income in the area. E.g., 5.0 = $50,000"
)

# House Age
house_age = st.sidebar.slider(
    "House Age (years)",
    min_value=1, max_value=52, value=20, step=1,
    help="Median age of houses in this block"
)

# Rooms
ave_rooms = st.sidebar.slider(
    "Average Rooms per House",
    min_value=1.0, max_value=15.0, value=6.0, step=0.1,
    help="Average number of rooms per household"
)

# Bedrooms
ave_bedrms = st.sidebar.slider(
    "Average Bedrooms per House",
    min_value=0.5, max_value=5.0, value=1.0, step=0.1,
    help="Average number of bedrooms per household"
)

# Population
population = st.sidebar.number_input(
    "Block Population",
    min_value=1, max_value=40000, value=1500, step=100,
    help="Total population in this geographic block"
)

# Occupancy
ave_occup = st.sidebar.slider(
    "Average Occupants per House",
    min_value=1.0, max_value=10.0, value=3.0, step=0.1,
    help="Average number of people living in each household"
)

st.sidebar.divider()
st.sidebar.subheader("📍 Location")

# Location - using selectbox for known cities (better UX than raw lat/long)
city_presets = {
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "San Diego": (32.7157, -117.1611),
    "Sacramento": (38.5816, -121.4944),
    "Fresno": (36.7378, -119.7871),
    "Custom (enter manually)": None
}

city_choice = st.sidebar.selectbox(
    "Select a City (or choose Custom)",
    options=list(city_presets.keys())
)

if city_presets[city_choice] is None:
    latitude = st.sidebar.slider(
        "Latitude", min_value=32.0, max_value=42.0, value=36.5, step=0.01
    )
    longitude = st.sidebar.slider(
        "Longitude", min_value=-125.0, max_value=-114.0, value=-119.5, step=0.01
    )
else:
    latitude, longitude = city_presets[city_choice]
    st.sidebar.info(f"📍 Lat: {latitude}, Long: {longitude}")
    
# Section 5 — Predict Button & Results Display
# ── MAIN AREA: PREDICTION ──────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    predict_clicked = st.button(
        "🔮 Predict House Price",
        type="primary",
        use_container_width=True
    )

if predict_clicked:
    input_data = {
        'MedInc'    : med_inc,
        'HouseAge'  : house_age,
        'AveRooms'  : ave_rooms,
        'AveBedrms' : ave_bedrms,
        'Population': population,
        'AveOccup'  : ave_occup,
        'Latitude'  : latitude,
        'Longitude' : longitude
    }

    try:
        with st.spinner("Calculating prediction..."):
            result = predictor.predict(input_data)

        st.success("✅ Prediction complete!")

        # ── Display results in metric cards ──
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                label="💰 Predicted Price",
                value=f"${result['predicted_price']:,.0f}"
            )

        with m2:
            st.metric(
                label="📉 Lower Estimate",
                value=f"${result['confidence_range'][0]:,.0f}"
            )

        with m3:
            st.metric(
                label="📈 Upper Estimate",
                value=f"${result['confidence_range'][1]:,.0f}"
            )

        st.caption(f"Model used: `{result['model_name']}`")

        st.divider()

        # ── Input Summary ──
        st.subheader("📋 Input Summary")
        summary_df = pd.DataFrame({
            'Feature': [
                'Median Income', 'House Age', 'Avg Rooms', 'Avg Bedrooms',
                'Population', 'Avg Occupants', 'Latitude', 'Longitude'
            ],
            'Value': [
                f"${med_inc * 10_000:,.0f}", f"{house_age} years",
                f"{ave_rooms:.1f}", f"{ave_bedrms:.1f}",
                f"{population:,}", f"{ave_occup:.1f}",
                f"{latitude:.2f}", f"{longitude:.2f}"
            ]
        })
        st.table(summary_df)

        # ── Map showing location ──
        st.subheader("📍 Location on Map")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_df, zoom=6)

    except ValueError as e:
        st.error(f"⚠️ Invalid input: {str(e)}")
    except Exception as e:
        st.error(f"❌ Something went wrong: {str(e)}")

else:
    st.info("👈 Adjust the house features in the sidebar, then click **Predict**.")
    
# Section 6 — Footer
# ── FOOTER ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        Built with Streamlit & Scikit-Learn | Data: California Housing Dataset (1990 Census)
        <br>This is a demo project for educational purposes — not financial advice.
    </div>
    """,
    unsafe_allow_html=True
)

