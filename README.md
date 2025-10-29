# 🐝 SustainaBee — *Small Actions, Big Hive Impact*

SustainaBee is a **smart sustainability dashboard** that helps users track, analyze, and simulate household resource usage — including **energy, water, gas, and waste** — while forecasting future consumption using **AI**.

It’s designed as a **prototype for a real IoT-based Smart Home system**, currently powered by synthetic data simulation, but easily extendable to **live sensor integrations** in the future.

---

## 🧠 Project Overview

SustainaBee transforms raw household data (real or simulated) into **meaningful insights, eco-scores, and cost forecasts**.  
The app allows users to:

- 📥 Upload their daily or monthly usage CSV  
- ⚙️ Simulate real-time smart sensor readings  
- 💰 Compute daily and monthly costs  
- 👨‍👩‍👧 Compare against family or community benchmarks  
- 🤖 View AI-driven predictions and eco insights  

This project demonstrates the **end-to-end structure of a sustainability platform** — from  
**data ingestion → cleaning → cost computation → AI insights** — in an interactive Streamlit dashboard.

---

## 🌿 Features at a Glance

| Module | Description |
|--------|--------------|
| 🏠 **Profile Setup** | Save your city, household size, and billing type for personalized cost rates. |
| 📥 **Data Upload / Simulation** | Upload CSVs or auto-generate synthetic sensor data for energy, water, gas, and waste. |
| 🧹 **Waste Bin Monitoring** | Tracks bin fill level (%) with alerts when nearing 80% and resets after reaching 100%. |
| 📊 **Daily Cost Summary (30 Days)** | Calculates daily usage cost using dynamic rate values fetched for your city. |
| 📅 **Historical Tracking** | Stores 30-day data for trend visualization and cost comparison. |
| ✍️ **Activity Logging** | Log eco-friendly or resource-using activities (like shower, cooking). |
| 🧭 **AI Forecasting (Demo)** | Uses a Random Forest model to simulate future energy cost predictions. |
| 💬 **AI Insights** | Generates sustainability tips, anomaly detections, and eco-reward points. |

---

## 🧩 App Flow (Simplified)

### 1️⃣ Data Ingestion
- Upload your `energy_usage.csv` file or click **Simulate Live Data**
- Columns:  
date, energy_kwh, water_liters, gas_m3, waste_kg, cost_inr, renewable_usage_kwh
🧠 *Future:* Replaced by **live IoT sensors** streaming data every few minutes.

---

### 2️⃣ Data Preparation
**Function:** `prepare_timeseries(df)`  
Cleans, validates, and aggregates uploaded readings:

- Removes duplicates  
- Fills missing data  
- Converts hourly readings → daily totals  
- Produces `summary_df` used across modules  

🧠 *Future:* Real-time streaming data cleaning & daily summarization.

---

### 3️⃣ Cost Calculation
**Functions:**  
`get_all_live_rates(city)` + `compute_costs(summary_df, rates)`

Dynamic city-based tariffs:
- ⚡ Electricity → ₹8.9/kWh  
- 💧 Water → ₹0.02/L  
- 🔥 Gas → ₹60/m³  
- 🗑️ Waste → ₹5/kg  

📊 **Displays:**
- Today’s total cost  
- 30-day trend graph  
- Weekly summaries  

🧠 *Future:* Real tariffs fetched dynamically from APIs (GHMC, BESCOM, etc.)

---

### 4️⃣ Family / Cohort Comparison
Compares your household with others of similar profile (city, home type, eco-score).  
🧠 *Future:* Uses **real IoT profiles** for live comparisons.

---

### 5️⃣ AI Forecasting & Insights
A pre-trained **Random Forest model** predicts next month’s usage.  
`generate_ai_insights()` and `detect_anomalies()` analyze patterns to:

- 🔺 Detect sudden spikes  
- 🔧 Suggest optimizations  
- 🌱 Reward eco-points for sustainable actions  

🧠 *Future:* Full ML pipeline trained on real IoT sensor logs.

---

### 6️⃣ Activity Logging
Log daily activities (e.g., “10 min shower”, “Cooking 30 min”).  
Stored in the SQLite database and displayed in the dashboard.

🧠 *Future:* Logged actions will dynamically adjust AI forecasts and eco-score.

---

### 7️⃣ Database (SQLite)
Backend database `app.db` stores:
- Daily sensor data  
- User profiles  
- Activity logs  

🧠 *Future:* Upgraded to **cloud sync (Firebase / Supabase)** for real-time access.

---

## 🧠 How It Works (Data Flow)

1️⃣ Upload or simulate data  
2️⃣ Clean and process readings  
3️⃣ Apply live rates → calculate cost  
4️⃣ Generate insights and forecasts  
5️⃣ Display analytics in dashboard  
6️⃣ *Future:* IoT sensors auto-stream usage data  
<img width="1908" height="848" alt="image" src="https://github.com/user-attachments/assets/e3d7bdd4-3391-45e5-aa99-53c43ec193dd" />


<img width="1501" height="723" alt="image" src="https://github.com/user-attachments/assets/a870bbd2-31c8-4d29-8bde-8f4e79c5c52f" />


<img width="1516" height="568" alt="image" src="https://github.com/user-attachments/assets/91094631-3445-42cf-869c-298285a98190" />

<img width="1470" height="783" alt="image" src="https://github.com/user-attachments/assets/ce49f3a1-baa5-4c9e-9298-8c564b2ccd65" />

<img width="1543" height="828" alt="image" src="https://github.com/user-attachments/assets/19d39c53-34d5-43f3-8e74-6b5537d82bea" />

<img width="1489" height="693" alt="image" src="https://github.com/user-attachments/assets/f798b148-a2ef-464b-9501-5dcbbb495f32" />

---
## 📂 Project Structure

SustainaBee/
│
├── app.py # Main Streamlit dashboard
├── utils.py # Backend logic: DB, AI, and cost functions
├── app.db # SQLite database (auto-created)
├── sample_data.csv # Example synthetic dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🗃️ Sample Data Format

| date | energy_kwh | water_liters | gas_m3 | waste_kg | cost_inr | renewable_usage_kwh |
|------|-------------|--------------|--------|-----------|-----------|----------------------|
| 2025-09-01 | 12.3 | 145 | 0.5 | 9.2 | 158.6 | 1.1 |
| 2025-09-02 | 11.9 | 151 | 0.4 | 8.7 | 153.2 | 1.3 |
| 2025-09-03 | 12.8 | 160 | 0.6 | 9.5 | 161.7 | 1.2 |

---

## 🧱 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Database** | SQLite |
| **ML** | Scikit-learn (RandomForestRegressor) |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Streamlit Charts, Plotly |

---

## 🔮 Future Enhancements

✅ Real-time IoT sensor integration  
✅ Family group dashboard comparison  
✅ Dynamic API tariff fetching  
✅ Cloud database sync  
✅ ML-based anomaly detection  
✅ Eco reward system  
✅ Nearby eco-store suggestions (Google Maps API)  
✅ Predictive alerts (energy/water/gas spike warnings)  
✅ Weather-based sustainability tips  
✅ Voice assistant integration (Alexa / Google Assistant)  
✅ Carbon footprint tracking & reporting  

---

## 💡 Motto
> “**Small Actions, Big Hive Impact** — because sustainability starts at home.” 🌍🐝


