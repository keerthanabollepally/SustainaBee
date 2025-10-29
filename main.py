import streamlit as st
import pandas as pd
import time
from utils1 import generate_live_sensor_data, generate_waste_data, load_sensor_csv

# ----------------------------------------------------------
# 🐝 App Setup
# ----------------------------------------------------------
st.set_page_config(page_title="SustainaBee", page_icon="🐝", layout="wide")

st.title("🐝 SustainaBee – Small Actions, Big Hive Impact")
st.markdown("### 🌱 Data Upload & Live Sensor Feed")

# Sidebar for selecting mode
st.sidebar.header("📦 Data Source")
mode = st.sidebar.radio("Choose Mode:", ["Upload CSV", "Live Sensor Feed (Demo)"])

# ----------------------------------------------------------
# MODE 1️⃣: Upload CSV
# ----------------------------------------------------------
if mode == "Upload CSV":
    uploaded = st.file_uploader("📁 Upload your sensor data (.csv)", type=["csv"])

    if uploaded:
        df = load_sensor_csv(uploaded)
        st.success("✅ Data Uploaded Successfully!")
        st.dataframe(df.head(10))

        st.markdown("### 📊 Summary")
        st.write(f"Total Records: **{len(df)}**")
        st.write("Sensor Types Detected:", ", ".join(df['type'].unique()))

    else:
        st.info("Upload a CSV file with columns: `timestamp`, `sensor_id`, `type`, `value`, `unit`")

# ----------------------------------------------------------
# MODE 2️⃣: Live Sensor Feed Simulation
# ----------------------------------------------------------
elif mode == "Live Sensor Feed (Demo)":
    st.markdown("### 📡 Simulated Real-Time Sensor Feed")

    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Simulation Duration (seconds):", 10, 180, 30)
    with col2:
        refresh_rate = st.slider("Update Every (seconds):", 1, 5, 3)

    placeholder = st.empty()
    waste_placeholder = st.empty()

    st.info("🕹️ Simulating Energy, Water, Gas, and Waste readings...")

    waste_level = 0
    waste_weight = 0.0

    for _ in range(duration // refresh_rate):
        df_live = generate_live_sensor_data()

        # Waste bin update
        waste_level, waste_weight, alert = generate_waste_data(waste_level, waste_weight)
        waste_row = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_id": "WS001",
            "type": "waste",
            "fill_level(%)": waste_level,
            "weight(kg)": round(waste_weight, 2),
            "alert": alert
        }

        df_live = pd.concat([df_live, pd.DataFrame([waste_row])], ignore_index=True)
        placeholder.dataframe(df_live)

        with waste_placeholder.container():
            st.subheader("🗑️ Waste Bin Fill Level")
            st.progress(min(waste_level / 100, 1.0))
            st.write(f"Fill Level: **{waste_level}%**, Weight: **{waste_weight:.2f} kg**")

            if alert:
                st.warning(alert)

        time.sleep(refresh_rate)

    st.success("✅ Simulation Complete!")
    st.balloons()
