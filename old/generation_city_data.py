# city_data_generator.py
import pandas as pd
import numpy as np
import random

def generate_city_data(num_rows_per_city=200, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    cities = ["Hyderabad", "Bengaluru", "Chennai", "Mumbai"]
    housing_types = ["Apartment", "Independent", "Villa"]

    all_data = []

    for city in cities:
        # tweak base patterns per city
        city_factor = {
            "Hyderabad": 1.0,
            "Bengaluru": 0.95,   # slightly more efficient
            "Chennai": 1.05,     # slightly higher usage
            "Mumbai": 1.1        # dense, high cost
        }[city]

        household_sizes = np.random.choice([2, 3, 4, 5, 6], num_rows_per_city, p=[0.15, 0.25, 0.3, 0.2, 0.1])
        housing_type = np.random.choice(housing_types, num_rows_per_city, p=[0.6, 0.3, 0.1])
        appliance_scores = np.round(np.random.uniform(0.5, 0.9, num_rows_per_city), 2)

        # daily kWh (city_factor scales usage)
        base_kWh = 10 + 1.2 * household_sizes + np.random.normal(0, 1.5, num_rows_per_city)
        efficiency_factor = 1 - (appliance_scores - 0.5)
        avg_daily_kWh = np.round(base_kWh * efficiency_factor * city_factor, 2)

        # renewable usage & monthly cost
        renewable_share = np.round(np.random.uniform(0.05, 0.25, num_rows_per_city), 2)
        renewable_kWh = np.round(avg_daily_kWh * renewable_share, 2)
        cost_inr = np.round(avg_daily_kWh * 30 * 11 * city_factor, 0)

        df_city = pd.DataFrame({
            "city": city,
            "household_size": household_sizes,
            "housing_type": housing_type,
            "appliances_score": appliance_scores,
            "avg_daily_kWh": avg_daily_kWh,
            "renewable_kWh": renewable_kWh,
            "monthly_cost_inr": cost_inr
        })

        all_data.append(df_city)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("city_energy_profiles.csv", index=False)
    print(f"✅ City dataset saved: city_energy_profiles.csv ({len(final_df)} rows across {len(cities)} cities)")

if __name__ == "__main__":
    generate_city_data()
