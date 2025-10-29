🐝 SustainaBee — Small Actions, Big Hive Impact

SustainaBee is a smart sustainability dashboard that helps users track, analyze, and simulate household resource usage — including energy, water, gas, and waste — while forecasting future consumption using AI.

It’s designed as a prototype for a real IoT-based Smart Home system, currently powered by synthetic data simulation, but easily extendable to live sensor integrations in the future.

🧠 Project Overview

SustainaBee transforms raw household data (real or simulated) into meaningful insights, eco-scores, and cost forecasts.
The app allows users to:

Upload their daily or monthly usage CSV

Simulate real-time smart sensor readings

Compute daily and monthly costs

Compare against family or community benchmarks

View AI-driven predictions and eco insights

⚡ The project demonstrates the end-to-end structure of a sustainability platform — from data ingestion → cleaning → cost computation → AI insights — in an interactive Streamlit dashboard.

🌿 Features at a Glance
Module	Description
🏠 Profile Setup	Save your city, household size, and billing type for personalized cost rates.
📥 Data Upload / Simulation	Upload your CSV or auto-generate live synthetic sensor data for energy, water, gas, and waste.
🧹 Waste Bin Monitoring	Tracks bin fill level (%) with alerts when nearing 80% and resets after reaching 100%.
📊 Daily Cost Summary (30 Days)	Calculates daily usage cost using dynamic rate values fetched for your city.
📅 Historical Tracking	Stores 30-day data for trend visualization and cost comparison.
✍️ Activity Logging	Lets users manually log eco-friendly or resource-using activities (like shower, cooking).
🧭 AI Forecasting (Demo)	Uses a Random Forest model to simulate future energy cost predictions.
💬 AI Insights	Generates sustainability tips, anomalies, and eco-reward points.
🧩 App Flow (Simplified)
1️⃣ Data Ingestion

Upload your energy_usage.csv file
(or use live simulation button)

Columns:

date, energy_kwh, water_liters, gas_m3, waste_kg, cost_inr, renewable_usage_kwh


🧠 In future: This step will be replaced by live IoT sensors automatically streaming usage data every few minutes.

2️⃣ Data Preparation

Function: prepare_timeseries(df)

Cleans, validates, and aggregates uploaded readings:

Removes duplicates

Fills missing data

Converts hourly readings → daily totals

Produces a clean table summary_df used across modules.

🧠 In future: Real-time streaming data will be cleaned and summarized daily in the background.

3️⃣ Cost Calculation

Function: get_all_live_rates(city) + compute_costs(summary_df, rates)

Applies dynamic city-based tariffs:

Electricity: ₹8.9/kWh

Water: ₹0.02/L

Gas: ₹60/m³

Waste: ₹5/kg

📊 Displays:

Today’s cost (₹)

30-day trend graph

Weekly summaries

🧠 In future: Real tariffs will be fetched from live APIs (e.g., GHMC, BESCOM, etc.)

4️⃣ Family / Cohort Comparison

Groups users into cohorts based on city, home type, or eco-score.

Compares your usage to similar households or family members.

🧠 In future: Uses real household profiles and IoT data from all family members.

5️⃣ AI Forecasting & Insights

A pre-trained Random Forest model predicts next month’s usage trend.

generate_ai_insights() and detect_anomalies() analyze patterns:

Detect sudden spikes 🔺

Suggest optimizations 🔧

Reward eco-points for low consumption 🌱

🧠 In future: Will use real machine learning on historical sensor logs.

6️⃣ Activity Logging

Log daily actions (e.g., “10 min shower”, “Cooking 30 min”).

Each entry is saved to the SQLite DB.

Currently displayed in “Recent Activity Log”.

🧠 In future: Logged actions will adjust AI predictions and eco-score dynamically.

7️⃣ Database (SQLite)

Backend database app.db handles:

Daily sensor data

User profiles

Logged activities

Auto-created when app runs.

🧠 In future: Will be upgraded to a cloud DB (Firebase / Supabase) for real-time updates.

🧠 How It Works (Flow)

1. Upload or simulate data

2. Clean and process data

3. Apply live rates → calculate cost

4. Generate insights and forecasts

5. Display in interactive dashboard

6. Future: Sensors push real-time updates

📂 Project Structure

SustainaBee/
│
├── app.py                 # Main Streamlit dashboard
├── utils.py               # Backend logic: DB, AI, and calculations
├── app.db                 # SQLite database (auto-generated)
├── sample_data.csv        # 1-day synthetic dataset example
├── requirements.txt       # Required Python dependencies
└── README.md              # Project documentation

🗃️ Sample Data Format| date       | energy_kwh | water_liters | gas_m3 | waste_kg | cost_inr | renewable_usage_kwh |
| ---------- | ---------- | ------------ | ------ | -------- | -------- | ------------------- |
| 2025-09-01 | 12.3       | 145          | 0.5    | 9.2      | 158.6    | 1.1                 |
| 2025-09-02 | 11.9       | 151          | 0.4    | 8.7      | 153.2    | 1.3                 |
| 2025-09-03 | 12.8       | 160          | 0.6    | 9.5      | 161.7    | 1.2                 |

🧱 Tech Stack

| Component         | Technology                           |
| ----------------- | ------------------------------------ |
| **Frontend**      | Streamlit                            |
| **Backend**       | Python                               |
| **Database**      | SQLite                               |
| **ML**            | Scikit-learn (RandomForestRegressor) |
| **Data Handling** | Pandas, NumPy                        |
| **Visualization** | Streamlit Charts, Plotly             |


🔮 Future Enhancements

✅ Real-time IoT sensor integration
✅ Family group dashboard comparison
✅ Dynamic API tariff fetching
✅ Cloud database sync
✅ ML-based anomaly detection
✅ Eco reward system
✅ Nearby eco-store suggestions via Google Maps API
✅ Predictive alerts (energy/water/gas spike warnings)
✅ Weather-based sustainability suggestions
✅ Voice assistant integration (e.g., Alexa/Google Assistant)
✅ Carbon footprint tracking and reporting
