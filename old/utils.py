import pandas as pd
import numpy as np
import sqlite3
import datetime
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# DATABASE HANDLING
# ==========================================================
class DB:
    def __init__(self, path="app.db"):
        self.path = path
        self._ensure_tables()

    def _conn(self):
        return sqlite3.connect(
            self.path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )

    def _ensure_tables(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    city TEXT,
                    household_size INTEGER,
                    housing_type TEXT,
                    appliances TEXT,
                    billing_type TEXT,
                    preferred_collection TEXT,
                    created_at TEXT
                )
            """)
            c.commit()

    def save_profile(self, profile: dict):
        with self._conn() as c:
            c.execute("""
                INSERT INTO profiles (
                    name, city, household_size, housing_type,
                    appliances, billing_type, preferred_collection, created_at
                ) VALUES (?,?,?,?,?,?,?,?)
            """, (
                profile['name'],
                profile['city'],
                profile['household_size'],
                profile['housing_type'],
                profile['appliances'],
                profile['billing_type'],
                profile['preferred_collection'],
                profile['created_at']
            ))
            c.commit()

    def list_profiles(self):
        with self._conn() as c:
            cur = c.execute("""
                SELECT name, city, household_size, housing_type,
                       appliances, billing_type, preferred_collection, created_at
                FROM profiles ORDER BY id DESC
            """)
            rows = cur.fetchall()
            keys = [
                "name", "city", "household_size", "housing_type",
                "appliances", "billing_type", "preferred_collection", "created_at"
            ]
            return [dict(zip(keys, r)) for r in rows]


# ==========================================================
# COST + RATES
# ==========================================================
def get_all_live_rates(city="Other"):
    """Simulated live tariff data — can later be replaced by API or scraper."""
    base = {
        "electricity_rate": 8.5,   # ₹/kWh
        "water_rate": 0.02,        # ₹/liter
        "lpg_rate": 110.0,         # ₹/kg
        "waste_rate": 5.0          # ₹/kg
    }

    if city.lower() in ["mumbai", "delhi", "bengaluru", "hyderabad"]:
        base["electricity_rate"] *= 1.05

    return base


def compute_costs(daily_df, rates):
    """Compute daily total cost for energy, water, waste, and gas."""
    df = daily_df.copy()
    for col in ["energy_kwh", "water_liters", "waste_kg", "gas_kg"]:
        if col not in df.columns:
            df[col] = 0

    df["energy_cost"] = df["energy_kwh"] * rates["electricity_rate"]
    df["water_cost"] = df["water_liters"] * rates["water_rate"]
    df["waste_cost"] = df["waste_kg"] * rates["waste_rate"]
    df["gas_cost"] = df["gas_kg"] * rates["lpg_rate"]
    df["total_cost"] = df[["energy_cost", "water_cost", "waste_cost", "gas_cost"]].sum(axis=1)

    return df


# ==========================================================
# SAMPLE DATA GENERATOR
# ==========================================================
def load_csv_sample():
    """Generate 30-day synthetic household usage sample."""
    today = datetime.date.today()
    dates = [today - datetime.timedelta(days=i) for i in range(30)][::-1]

    rows = []
    for d in dates:
        rows.append({
            "date": pd.to_datetime(d),
            "user": "demo",
            "energy_kwh": max(0.5, np.random.normal(6, 1.8)),
            "water_liters": max(50, np.random.normal(200, 60)),
            "waste_kg": max(0.2, np.random.normal(1.2, 0.6)),
            "gas_kg": max(0.0, np.random.normal(0.2, 0.2)),
        })

    return pd.DataFrame(rows)


# ==========================================================
# TIMESERIES PREPARATION
# ==========================================================
def prepare_timeseries(df):
    """Normalize and aggregate uploaded energy dataset for eco analytics."""
    df.columns = df.columns.str.strip().str.lower()

    if "date" not in df.columns:
        raise ValueError("❌ 'date' column not found in dataset!")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.date

    # Handle uploaded CSV like your example (energy_kWh, Cost_INR, Renewable_Usage_kWh)
    if "cost_inr" in df.columns and "renewable_usage_kwh" in df.columns:
        df.rename(columns={"renewable_usage_kwh": "renewablekwh"}, inplace=True)
        df["energy_kwh"] = df.get("energy_kwh", df.get("energykwh", 0))
        df["water_liters"] = np.random.uniform(150, 250, len(df))
        df["waste_kg"] = np.random.uniform(1, 2, len(df))
        df["gas_kg"] = np.random.uniform(0.1, 0.3, len(df))
    else:
        for col in ["energy_kwh", "water_liters", "waste_kg", "gas_kg"]:
            if col not in df.columns:
                df[col] = 0.0

    ag = df.groupby("date")[["energy_kwh", "water_liters", "waste_kg", "gas_kg"]].sum().reset_index()

    ag["eco_score"] = 100 - (
        0.4 * (ag["energy_kwh"] / ag["energy_kwh"].max()) * 100 +
        0.3 * (ag["water_liters"] / ag["water_liters"].max()) * 100 +
        0.2 * (ag["waste_kg"] / ag["waste_kg"].max()) * 100 +
        0.1 * (ag["gas_kg"] / ag["gas_kg"].max()) * 100
    )
    ag["eco_score"] = ag["eco_score"].clip(lower=0, upper=100)
    return ag


# ==========================================================
# MODEL LOADING (optional)
# ==========================================================
def load_model_if_present(model_file):
    """Load serialized model (.pkl) if provided."""
    if model_file is None:
        return None
    try:
        obj = pickle.load(model_file)
        return obj
    except Exception as e:
        print("Model load failed:", e)
        return None


# ==========================================================
# PHASE B – PERSONALIZATION & BENCHMARKING
# ==========================================================
def load_city_profiles(path="city_energy_profiles.csv"):
    """Load synthetic or real multi-city benchmark dataset."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df


def build_cohorts(city_df, n_clusters=5):
    """Cluster households using K-Means to form peer cohorts."""
    features = ["household_size", "appliances_score", "avg_dailykwh", "renewablekwh"]
    X = city_df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    city_df["cluster"] = kmeans.fit_predict(X_scaled)

    return city_df, kmeans, scaler


def compare_user_to_cohort(user_profile, city_df, kmeans, scaler):
    """Compare a single user’s profile with cluster (peer group) averages."""
    feats = ["household_size", "appliances_score", "avg_dailykwh", "renewablekwh"]

    user_df = pd.DataFrame([user_profile])[feats]
    user_scaled = scaler.transform(user_df)
    cluster_id = kmeans.predict(user_scaled)[0]

    cohort_stats = city_df.groupby("cluster")[feats + ["monthly_cost_inr"]].mean().reset_index()
    row = cohort_stats.loc[cohort_stats["cluster"] == cluster_id].iloc[0]

    comparison = {
        "cluster_id": int(cluster_id),
        "avg_dailykwh": float(row["avg_dailykwh"]),
        "renewablekwh": float(row["renewablekwh"]),
        "monthly_cost_inr": float(row["monthly_cost_inr"]),
        "relative_usage_pct": round((user_profile["avg_dailykwh"] / row["avg_dailykwh"]) * 100, 1),
        "relative_cost_pct": round((user_profile["monthly_cost_inr"] / row["monthly_cost_inr"]) * 100, 1)
    }

    return comparison
def generate_ai_insights(user_df, city_df):
    """Generate GPT-style insights comparing household to peers."""
    row = user_df.iloc[-1]  # last user entry
    avg_usage = city_df["avg_dailykwh"].mean()
    avg_renewable = city_df["renewablekwh"].mean()

    usage = row["avg_dailykwh"]
    renewable = row["renewablekwh"]

    # --- Ratings ---
    efficiency_score = (renewable / usage) * 100 if usage else 0
    if efficiency_score >= 70:
        rating = "⭐⭐⭐⭐⭐ (Excellent)"
    elif efficiency_score >= 50:
        rating = "⭐⭐⭐⭐ (Good)"
    elif efficiency_score >= 30:
        rating = "⭐⭐⭐ (Average)"
    else:
        rating = "⭐⭐ (Needs improvement)"

    # --- Feedback ---
    if usage > avg_usage:
        usage_feedback = (
            f"Your average daily usage ({usage:.1f} kWh) is **{usage - avg_usage:.1f} kWh higher** "
            f"than your city's average. Try checking heavy-load appliances or reducing A/C time."
        )
    else:
        usage_feedback = (
            f"Nice! Your usage ({usage:.1f} kWh) is **{avg_usage - usage:.1f} kWh lower** "
            f"than the city average. You're managing energy well!"
        )

    if renewable < avg_renewable:
        renewable_feedback = (
            "You’re using less renewable energy than peers. Consider installing solar panels "
            "or subscribing to green energy options if available."
        )
    else:
        renewable_feedback = (
            "You’re above average in renewable usage — keep it up!"
        )

    summary = f"""
### 🌍 AI Insight Summary

**Efficiency Rating:** {rating}

**Usage Analysis:**  
{usage_feedback}

**Renewable Energy:**  
{renewable_feedback}
"""

    return summary


# ==========================================================
# PHASE D – APPLIANCE USAGE, REWARDS & ECO STORES
# ==========================================================

def generate_appliance_usage(appliances):
    """
    Estimate per-appliance daily energy consumption (synthetic demo values).
    Later you can link this to real meter readings or appliance databases.
    """
    default_usage = {
        "AC": 4.5,
        "Fridge": 1.1,
        "Geyser": 2.8,
        "WashingMachine": 0.9,
        "Induction": 1.5,
        "Dishwasher": 1.3,
        "TV": 0.6,
        "Solar": -1.5  # renewable contribution
    }

    data = []
    for a in appliances:
        val = default_usage.get(a, np.random.uniform(0.3, 1.2))
        data.append({
            "appliance": a,
            "daily_kwh": round(val, 2),
            "tips": _appliance_tip(a)
        })

    return pd.DataFrame(data)


def _appliance_tip(appliance):
    """Return a simple energy-saving tip for each appliance."""
    tips = {
        "AC": "Set at 26 °C, clean filters monthly, and use timers.",
        "Fridge": "Defrost weekly and keep 3 inches gap from wall.",
        "Geyser": "Use only 10–15 minutes daily and insulate tank.",
        "WashingMachine": "Use eco mode & full loads.",
        "Induction": "Use flat-bottom cookware for higher efficiency.",
        "Dishwasher": "Run only full loads and use air-dry mode.",
        "TV": "Turn off completely when not watching.",
        "Solar": "Clean panels regularly for max output."
    }
    return tips.get(appliance, "Use efficiently and unplug when not in use.")


def calculate_rewards(comparison, user_profile):
    """
    Simple point system based on efficiency and renewable usage.
    Returns a numeric score (0–500 range).
    """
    points = 0
    if comparison.get("relative_usage_pct", 100) < 100:
        points += 20
    if user_profile.get("renewablekwh", 0) > 1.5:
        points += 30
    if comparison.get("relative_usage_pct", 100) < 90:
        points += 50
    points += 5  # daily login
    return min(points, 500)


def load_eco_stores(city="Hyderabad"):
    """
    Load sample eco-friendly store data for given city.
    Later this can be replaced by a live Google Maps / Places API lookup.
    """
    stores = pd.DataFrame({
        "city": ["Hyderabad", "Hyderabad", "Bengaluru", "Chennai"],
        "store_name": ["EcoSmart Store", "GreenLeaf Mart", "SolarX Hub", "EarthSaver"],
        "category": ["LEDs & Appliances", "Eco Groceries", "Solar Equipment", "Recycled Goods"],
        "offer": ["5 % off", "10 % off", "₹500 off", "15 % off"],
        "points_needed": [100, 150, 300, 200]
    })

    return stores[stores["city"].str.lower() == city.lower()]

# ------------------------------------------------
# Anomaly detection (simple z-score logic)
# ------------------------------------------------
def detect_anomalies(df, columns=['energy_kwh', 'water_liters', 'waste_kg', 'gas_kg'], threshold=2.0):
    df_anom = df.copy()
    for col in columns:
        mean, std = df_anom[col].mean(), df_anom[col].std()
        df_anom[f'{col}_anomaly'] = abs(df_anom[col] - mean) > (threshold * std)
    return df_anom

# ------------------------------------------------
# User-level personalized nudge
# ------------------------------------------------
def generate_user_nudge(row):
    messages = []
    if row.get('energy_kwh_anomaly', False):
        messages.append("⚠️ Energy spike detected. Check appliance usage or turn off unused devices.")
    if row.get('water_liters_anomaly', False):
        messages.append("⚠️ Water usage is unusually high. Try reducing shower time or check for leaks.")
    if row.get('waste_kg_anomaly', False):
        messages.append("⚠️ Higher waste generated this week. Can you segregate more or compost food scraps?")
    if row.get('gas_kg_anomaly', False):
        messages.append("⚠️ Gas usage high. Review cooking patterns or schedule a refill soon.")
    if not messages:
        messages.append("✅ All resource usage within normal range – great job!")
    return ' '.join(messages)

# ------------------------------------------------
# Granular activity logging (example schema)
# ------------------------------------------------
def log_activity(df, date, user, activity, resource_used, duration_minutes):
    act = {
        'date': date,
        'user': user,
        'activity': activity,
        'resource_used': resource_used,
        'duration_minutes': duration_minutes
    }
    return pd.concat([df, pd.DataFrame([act])], ignore_index=True)


# ------------------------------------------------
# Family/group challenge suggestion
# ------------------------------------------------
def suggest_family_challenge(df):
    total_water = df['water_liters'].sum()
    total_energy = df['energy_kwh'].sum()
    challenge_msg = (
        f"This week, try to reduce household water use by 10% (goal: {0.9*total_water:.0f} L) "
        f"and energy by 8% (goal: {0.92*total_energy:.1f} kWh) compared to your current average."
    )
    return challenge_msg

