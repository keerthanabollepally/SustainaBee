import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import os

from utils import (
    DB,
    load_city_profiles,
    build_cohorts,
    compare_user_to_cohort,
    prepare_timeseries,
    compute_costs,
    get_all_live_rates,
    load_csv_sample,
    generate_ai_insights,
    generate_appliance_usage,
    calculate_rewards,
    load_eco_stores,
    detect_anomalies,
    generate_user_nudge,
    log_activity,
    suggest_family_challenge
)

# =========================================
# PAGE CONFIG + THEME (black glass, neon green accent)
# =========================================
st.set_page_config(
    page_title="SustainaBee — Small Actions, Big Hive Impact 🐝🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS for glass-dark look and neon accents
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #050505 100%);
        color: #E6F9EA;
    }

    /* Header */
    .sustaina-header {
        text-align:center;
        padding: 6px 0 14px 0;
    }
    .sustaina-header h1 {
        font-family: 'Poppins', sans-serif;
        color: #00FF99;
        font-size: 2.4rem;
        margin: 0;
    }
    .sustaina-header p {
        color: #A8DFA3;
        margin: 2px 0 10px 0;
    }

    /* Sidebar (glass) */
    section[data-testid="stSidebar"] {
        background: rgba(10,10,10,0.62);
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(0,255,153,0.08);
        color: #E6F9EA;
    }
    section[data-testid="stSidebar"] .css-1d391kg { color: #E6F9EA; }

    /* Expander header styling */
    .streamlit-expanderHeader {
        color: #E6F9EA;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #00FF99;
        color: #000;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Metric values */
    .stMetricValue {
        color: #00FF99 !important;
    }

    /* DataFrame background */
    .stDataFrame, .stTable {
        color: #E6F9EA;
    }

    /* Keep font neat */
    * { font-family: "Poppins", sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================
# HEADER (centered SustainaBee)
# =========================================
st.markdown(
    """
    <div class="sustaina-header">
        <h1>🐝 SustainaBee</h1>
        <p>🍃 <em>Small Actions, Big Hive Impact</em></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================
# DATABASE (persistent profiles)
# =========================================
db = DB("app.db")

# =========================================
# SIDEBAR: Profile
# =========================================
st.sidebar.header("👤Profile")

with st.sidebar.form("profile_form", clear_on_submit=False):
    name = st.text_input("Name", value="Keerthana")
    city = st.selectbox(
        "City / Region",
        ["Kondapur", "Hyderabad", "Mumbai", "Delhi", "Bengaluru", "Other"]
    )
    household_size = st.number_input("Household size", min_value=1, max_value=10, value=3)
    housing_type = st.radio("Housing type", ["1BHK", "2BHK", "3BHK", "House"])
    appliances = st.multiselect(
        "Appliances (select all that apply)",
        ["AC", "Fridge", "Geyser", "WashingMachine", "Induction", "Dishwasher", "TV", "Solar"]
    )
    billing_type = st.selectbox("Electricity billing type", ["flat", "slab", "TOU", "unknown"])
    preferred_collection = st.selectbox(
        "Waste collection day (optional)",
        ["None", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    submitted = st.form_submit_button("Save profile")

if submitted:
    profile = dict(
        name=name,
        city=city,
        household_size=int(household_size),
        housing_type=housing_type,
        appliances=";".join(appliances),
        billing_type=billing_type,
        preferred_collection=preferred_collection,
        created_at=str(datetime.date.today())
    )
    db.save_profile(profile)
    st.sidebar.success("Profile saved ✅")

profiles = db.list_profiles()
st.sidebar.markdown("**Saved profiles:**")
for p in profiles:
    st.sidebar.write(f"- {p['name']} ({p['city']}, {p['housing_type']}, size {p['household_size']})")

# =========================================
# Session-state persistent objects
# =========================================
if "df" not in st.session_state:
    st.session_state.df = load_csv_sample()
df = st.session_state.df

# ✅ MODEL PATH — update this as needed
MODEL_PATH = r"C:\Users\SAI CHARAN RAJU\OneDrive\Desktop\personalized_ecocoach\eco_predictor_model.pkl"

# ✅ MODEL LOADING (correct handling)
if "model" not in st.session_state:
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            st.session_state.model = model
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.session_state.model = None
            st.sidebar.error(f"⚠️ Error loading model: {e}")
    else:
        st.session_state.model = None
        st.sidebar.warning("⚠️ Model file not found at the given path.")
model = st.session_state.model


# =========================================
# MAIN: Expandable single-page layout
# Each major section is an expander (collapsed by default except Data Ingestion)
# =========================================

# ---------- DATA INGESTION ----------
with st.expander("📂 Data Hive: Import Your Footprint", expanded=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(
            "Upload your merged dataset CSV (optional). "
            "**Expected columns:** `date,user,energy_kwh,water_liters,weight_kg,usage_kg,eco_score`"
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            try:
                df_uploaded = pd.read_csv(uploaded, parse_dates=["date"])
                st.session_state.df = df_uploaded
                df = st.session_state.df
                st.success("✅ CSV uploaded and loaded.")
            except Exception:
                df_uploaded = pd.read_csv(uploaded)
                st.session_state.df = df_uploaded
                df = st.session_state.df
                st.warning("⚠ Uploaded CSV loaded (dates not parsed).")

        else:
            st.info("No CSV uploaded — using demo sample dataset.")
            # df already has default sample

        st.write("**Preview:**")
        st.dataframe(df.head())

    with col2:
        st.markdown("**Quick actions**")
        if st.button("Reload demo sample"):
            st.session_state.df = load_csv_sample()
            df = st.session_state.df
            st.success("Demo sample loaded.")
        if st.button("Clear uploaded dataset"):
            st.session_state.df = load_csv_sample()
            df = st.session_state.df
            st.info("Dataset reset to demo sample.")

# ---------- RATES block (small) ----------
with st.expander("💱  Live Buzz: Today's Energy & Water Rates", expanded=False):
    try:
        rates = get_all_live_rates(city if 'city' in locals() else "Other")
    except Exception:
        rates = {"electricity_rate": 8.5, "water_rate": 1.2}
    st.write("Simulated live rates (used for cost calculations):")
    st.json(rates)

# ---------- DASHBOARD ----------
with st.expander("📊 Hive Dashboard: Your Daily Impact", expanded=False):
    try:
        # compute summaries
        summary_df = prepare_timeseries(df)
        costs_df = compute_costs(summary_df, rates)

        # safe access to today's row
        if "date" in summary_df.columns:
            latest_date = summary_df['date'].max()
            today_row = summary_df[summary_df['date'] == latest_date].iloc[0]
        else:
            today_row = summary_df.iloc[-1]

        # show metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        try:
            mcol1.metric("Energy today (kWh)", f"{today_row.energy_kwh:.2f}",
                         delta=f"{(today_row.energy_kwh - summary_df.energy_kwh.mean()):+.2f}")
        except Exception:
            mcol1.metric("Energy today (kWh)", "N/A")
        try:
            mcol2.metric("Water today (L)", f"{today_row.water_liters:.0f}")
        except Exception:
            mcol2.metric("Water today (L)", "N/A")
        try:
            mcol3.metric("Waste this week (kg)", f"{summary_df.tail(7).waste_kg.sum():.2f}")
        except Exception:
            mcol3.metric("Waste this week (kg)", "N/A")
        try:
            today_cost = costs_df.loc[costs_df.date == latest_date, 'total_cost'].values[0]
            mcol4.metric("Today cost (₹)", f"{today_cost:.2f}")
        except Exception:
            mcol4.metric("Today cost (₹)", "N/A")

        # trends
        st.subheader("Trends — last 30 days")
        try:
            last30 = summary_df.tail(30).set_index("date")
            st.line_chart(last30[["energy_kwh", "water_liters"]])
        except Exception:
            st.info("Not enough time-series data for 30-day trends.")

    except Exception as e:
        st.error(f"Dashboard failed to render: {e}")

# ---------- WHAT-IF SIMULATOR (inline in dashboard) ----------
with st.expander("🔧 Eco Actions Playground: What If...?", expanded=False):
    try:
        reduce_shower_min = st.slider("Reduce average shower by (minutes/day)", 0, 10, 1)
        reduce_ac_hours = st.slider("Reduce AC usage (hours/day total)", 0.0, 5.0, 0.5)

        L_per_min_shower = 7.5
        kwh_per_ac_hour_per_unit = 1.2
        ac_count = max(1, sum([1 for a in appliances if a == "AC"])) if 'appliances' in locals() else 1
        monthly_days = 30

        saved_liters = reduce_shower_min * L_per_min_shower * monthly_days
        saved_kwh = reduce_ac_hours * kwh_per_ac_hour_per_unit * ac_count * monthly_days
        try:
            saved_money = saved_kwh * rates["electricity_rate"] + (saved_liters / 1000.0) * rates["water_rate"]
        except Exception:
            saved_money = saved_kwh * 8.5 + (saved_liters / 1000.0) * 1.2

        st.write(f"Estimated monthly savings: **~{saved_liters:.0f} L water**, **{saved_kwh:.1f} kWh energy** → **~₹{saved_money:.0f}/month**")
    except Exception as e:
        st.error(f"What-if simulator error: {e}")

# ---------- FORECAST ----------
with st.expander("🔮 Bee-Forecast: Next Month's Outlook", expanded=False):
    try:
        # ensure summary_df and costs_df exist
        if 'summary_df' not in locals():
            summary_df = prepare_timeseries(df)
            costs_df = compute_costs(summary_df, rates)

        if model is not None:
            st.success("✅ Model loaded — preparing features for prediction.")
            feature_pool = {
                "cost_in_rupees": costs_df["total_cost"].mean() if "total_cost" in costs_df.columns else 0,
                "eco_score": df["eco_score"].mean() if "eco_score" in df.columns else 7.0,
                "energy_kwh": summary_df["energy_kwh"].mean(),
                "waste_weight": summary_df["waste_kg"].mean() if "waste_kg" in summary_df.columns else 0,
                "water_liters": summary_df["water_liters"].mean(),
                "mean_energy_30": summary_df.tail(30)["energy_kwh"].mean() if len(summary_df) >= 1 else summary_df["energy_kwh"].mean(),
                "mean_water_30": summary_df.tail(30)["water_liters"].mean() if len(summary_df) >= 1 else summary_df["water_liters"].mean(),
                "household_size": household_size
            }

            expected_features = getattr(model, "feature_names_in_", list(feature_pool.keys()))
            feat_vector = {f: feature_pool.get(f, 0.0) for f in expected_features}
            X = pd.DataFrame([feat_vector])[expected_features]
            st.write("🧩 Prepared features for prediction:")
            st.dataframe(X.T.rename(columns={0: "value"}))

            try:
                pred = model.predict(X)[0]
                st.metric("Predicted energy next month (kWh)", f"{pred:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.info("No model found — predictions disabled. Upload your model to the configured MODEL_PATH to enable predictions.")
            # optionally show a simple simulated forecast
            sim_forecast = summary_df["energy_kwh"].mean() * np.random.uniform(0.95, 1.08)
            st.metric("Simulated predicted energy (kWh)", f"{sim_forecast:.2f}")
    except Exception as e:
        st.error(f"Forecast section error: {e}")

# ---------- PERSONALIZATION & COHORTS ----------
with st.expander("👥 Hive Mates: Compare & Personalize", expanded=False):
    CITY_DATA_PATH = r"C:\Users\SAI CHARAN RAJU\OneDrive\Desktop\personalized_ecocoach\city_energy_profiles.csv"
    try:
        if os.path.exists(CITY_DATA_PATH):
            city_df = load_city_profiles(CITY_DATA_PATH)
            city_df, kmeans, scaler = build_cohorts(city_df)

            user_profile = {
                "household_size": int(household_size),
                "appliances_score": len(appliances) * 1.5,
                "avg_dailykwh": summary_df["energy_kwh"].mean(),
                "renewablekwh": np.random.uniform(0.5, 2.5),
                "monthly_cost_inr": costs_df["total_cost"].sum() if "total_cost" in costs_df else 0
            }

            comparison = compare_user_to_cohort(user_profile, city_df, kmeans, scaler)
            st.success("✅ City cohort data loaded.")
            st.write("Cohort comparison results:")
            st.json(comparison)

            rel_usage = comparison.get('relative_usage_pct', 100.0)
            if rel_usage > 110:
                feedback = f"⚠ You consume {rel_usage:.1f}% more energy than your cohort. Try reducing AC usage or switching to efficient lighting."
            elif rel_usage < 90:
                feedback = f"🌱 Great job! You're {100 - rel_usage:.1f}% below your cohort's usage — keep it up!"
            else:
                feedback = f"✅ You’re roughly on par with similar homes in {city}."

            st.info(feedback)

            # AI insights
            try:
                user_df_for_ai = pd.DataFrame([{
                    "avg_dailykwh": user_profile["avg_dailykwh"],
                    "renewablekwh": user_profile["renewablekwh"]
                }])
                ai_insights = generate_ai_insights(user_df_for_ai, city_df)
                st.subheader("AI Insights")
                st.markdown(ai_insights)
            except Exception as e:
                st.error(f"Phase C AI insights failed: {e}")
        else:
            st.warning(f"City dataset not found at {CITY_DATA_PATH}. Please check path.")
    except Exception as e:
        st.error(f"Personalization failed: {e}")

# ---------- APPLIANCE USAGE + REWARDS + ECO STORES ----------
with st.expander("💡 Impact & Rewards: Devices & Bee-nefits", expanded=False):
    try:
        st.subheader("🔌 Appliance-wise Usage Breakdown")
        appliance_data = pd.DataFrame({
            "appliance": ["AC", "Refrigerator", "Lighting", "Washing Machine", "TV"],
            "daily_kwh": [4.2, 1.1, 0.8, 0.9, 0.5],
            "tips": [
                "Set AC to 26 °C and clean filters monthly.",
                "Defrost fridge weekly and keep it 3 in from wall.",
                "Switch to LED bulbs and use daylight where possible.",
                "Run washing machine on eco mode.",
                "Turn off TV when not watching."
            ]
        })
        st.bar_chart(appliance_data.set_index("appliance")["daily_kwh"])
        for _, row in appliance_data.iterrows():
            st.write(f"**{row['appliance']}** — {row['tips']}")

        # Rewards
        st.subheader("🎯 Eco Reward Points")
        points = 0
        # guard usage of comparison/user_profile
        try:
            if comparison.get("relative_usage_pct", 100) < 100:
                points += 20
        except Exception:
            pass
        try:
            if user_profile.get("renewablekwh", 0) > 1.5:
                points += 30
        except Exception:
            pass
        try:
            if comparison.get("relative_usage_pct", 100) < 90:
                points += 50
        except Exception:
            pass
        points += 5  # daily login

        st.metric("🌟 Eco Points", f"{points} pts")
        st.progress(min(points / 500, 1.0))
        st.caption("Earn points by saving energy and water! 500 pts = unlock coupon tiers.")

        # Nearby stores
        st.subheader("🛍 Nearby Eco-Friendly Stores")
        try:
            user_city = city if 'city' in locals() else "Hyderabad"
            city_stores = load_eco_stores(user_city)
        except Exception:
            city_stores = pd.DataFrame({
                "city": ["Hyderabad", "Hyderabad", "Bengaluru", "Chennai"],
                "store_name": ["EcoSmart Store", "GreenLeaf Mart", "SolarX Hub", "EarthSaver"],
                "category": ["LEDs & Appliances", "Eco Groceries", "Solar Equipment", "Recycled Goods"],
                "offer": ["5 % off", "10 % off", "₹500 off", "15 % off"],
                "points_needed": [100, 150, 300, 200]
            })

        if not city_stores.empty:
            st.write(f"Showing eco-friendly stores near *{user_city}* 🌿")
            for _, row in city_stores.iterrows():
                st.write(f"**{row['store_name']}** — {row.get('offer','')} (Need {row.get('points_needed', 999)} pts)")
                st.progress(min(points / max(row.get('points_needed', 1), 1), 1.0))
                if points >= row.get('points_needed', 999):
                    st.success("✅ Offer Unlocked!")
                else:
                    st.warning(f"🔒 Need {row.get('points_needed', 999) - points} more pts to unlock.")
        else:
            st.info(f"No eco stores found near *{user_city}* yet.")

    except Exception as e:
        st.error(f"Appliance/Rewards/Stores section failed: {e}")

# ---------- LOGS, ACTIVITY & EXPORT ----------
with st.expander("📘  Activity Book & Bee Challenges", expanded=False):
    try:
        # Show daily costs
        try:
            st.subheader("📅 Daily costs (last 30 days)")
            st.dataframe(costs_df.tail(30))
        except Exception:
            st.info("Costs data not available yet. Please upload dataset in Data Ingestion.")

        # Anomaly detection & nudge
        try:
            summary_df = detect_anomalies(summary_df)
            latest = summary_df.iloc[-1]
            nudge_msg = generate_user_nudge(latest)
            st.info(nudge_msg)
        except Exception:
            pass

        # Activity log (Correct scope/indentation: ALL inside try block)
        if "activity_df" not in st.session_state:
            st.session_state['activity_df'] = pd.DataFrame(
                columns=['date', 'user', 'activity', 'resource_used', 'duration_minutes']
            )

        st.subheader("🧾 Log individual activities")
        activity_options = [
            "Shower", "Cooking", "Laundry", "Dishwashing", "Room Cleaning",
            "Plant Watering", "AC Use", "TV Use", "Ironing Clothes",
            "Charging Devices", "Grocery Shopping", "Recycling", "Composting", "Other"
        ]
        activity = st.selectbox("Choose an activity", activity_options)
        resource_used = st.selectbox("Resource", ['water', 'energy', 'gas', 'waste'])
        duration_minutes = st.number_input("Duration (minutes)", min_value=1, max_value=180, value=10)
        log_activity_button = st.button("Log Activity")

        if log_activity_button:
            logged = log_activity(
                st.session_state['activity_df'],
                datetime.date.today(),
                name if 'name' in locals() else "User",
                activity,
                resource_used,
                duration_minutes
            )
            st.session_state['activity_df'] = logged
            st.success("✅ Activity logged!")

        st.write("Recent Activity Log:")
        st.dataframe(st.session_state['activity_df'].tail(10))

        # Family challenge
        st.subheader("🏡 Suggest family/group challenge")
        family_members = [p['name'] for p in profiles] if profiles else []
        challenge = suggest_family_challenge(summary_df if 'summary_df' in locals() else df)
        st.warning(challenge)

        # Export
        if st.button("📤 Export last 30 days to CSV"):
            try:
                tmp = costs_df.tail(30)
                tmp.to_csv("export_last30.csv", index=False)
                st.success("✅ Wrote export_last30.csv to your current folder.")
            except Exception as e:
                st.error(f"Export failed: {e}")

    except Exception as e:
        st.error(f"Logs/Export section failed: {e}")


# Final small footer
st.markdown("---")
st.markdown("<small style='color:#9EAFA0'>Built with ❤️ — SustainaBee</small>", unsafe_allow_html=True)
