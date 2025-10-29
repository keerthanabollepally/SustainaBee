import pandas as pd
import numpy as np
import datetime
import random

# ----------------------------------------------------------
# 1️⃣ Load Uploaded CSV
# ----------------------------------------------------------
def load_sensor_csv(file):
    df = pd.read_csv(file)
    required_cols = ["timestamp", "sensor_id", "type", "value", "unit"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

# ----------------------------------------------------------
# 2️⃣ Generate Synthetic Live Sensor Data
# ----------------------------------------------------------
def generate_live_sensor_data():
    """
    Simulate live readings for Energy, Water, Gas sensors.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sensors = [
        {"sensor_id": "E001", "type": "energy", "value": round(np.random.uniform(0.5, 2.5), 2), "unit": "kWh"},
        {"sensor_id": "W001", "type": "water", "value": round(np.random.uniform(0.2, 1.2), 2), "unit": "L"},
        {"sensor_id": "G001", "type": "gas", "value": round(np.random.uniform(0.01, 0.1), 3), "unit": "m³"},
    ]

    for s in sensors:
        s["timestamp"] = now
        s["device_status"] = random.choice(["active", "idle"])
        s["battery"] = random.randint(75, 100)
        s["location"] = random.choice(["kitchen", "bathroom", "balcony"])

    df = pd.DataFrame(sensors)
    return df[["timestamp", "sensor_id", "type", "value", "unit", "device_status", "battery", "location"]]

# ----------------------------------------------------------
# 3️⃣ Waste Bin Simulation
# ----------------------------------------------------------
def generate_waste_data(prev_level, prev_weight):
    """
    Simulate waste fill increase over time.
    - Fill increases slowly.
    - Resets when full.
    """
    increase = random.randint(2, 10)  # % per update
    new_level = prev_level + increase
    new_weight = prev_weight + (increase * 0.05)

    alert = None
    if new_level >= 100:
        new_level = 0
        new_weight = 0
        alert = "🧹 Waste collected – bin reset!"
    elif new_level >= 80:
        alert = "⚠️ Bin nearly full!"

    return new_level, new_weight, alert
