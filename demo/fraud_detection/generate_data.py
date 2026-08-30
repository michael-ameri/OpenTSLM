
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_fraud_data(output_path="demo/fraud_detection/fraud_data.csv", num_series=100):
    """
    Generates dummy mobile carrier data for fraud detection.
    """

    # Constants
    DAYS = 7
    INTERVAL_MINUTES = 15
    STEPS_PER_DAY = (24 * 60) // INTERVAL_MINUTES
    TOTAL_STEPS = DAYS * STEPS_PER_DAY

    data = []

    print(f"Generating {num_series} time series with {TOTAL_STEPS} steps each...")

    start_date = datetime(2026, 2, 2, 9, 15)

    for series_id in range(num_series):
        # Base random walk for features
        revenue = np.abs(np.cumsum(np.random.randn(TOTAL_STEPS))) * 10
        cost = revenue * 0.4 + np.random.randn(TOTAL_STEPS) * 2
        call_duration = np.abs(np.random.randn(TOTAL_STEPS) * 5 + 2)

        # Determine if this series will have a TPA trigger
        has_trigger = random.random() < 0.8 # 80% have a trigger

        # Determine if it is actual fraud (only if triggered)
        is_fraud = False
        if has_trigger:
            is_fraud = random.random() < 0.5

        tpa_values = [np.nan] * TOTAL_STEPS

        # Inject patterns
        if has_trigger:
            # Trigger usually happens at the end
            trigger_idx = TOTAL_STEPS - 1
            tpa_values[trigger_idx] = 608 # Example rule ID

            if is_fraud:
                # Fraud pattern: Sudden spike in revenue and cost before trigger
                spike_start = max(0, trigger_idx - 10)
                revenue[spike_start:trigger_idx+1] += 500
                cost[spike_start:trigger_idx+1] += 100
                call_duration[spike_start:trigger_idx+1] += 20

        # Generate timestamps
        timestamps = [start_date + timedelta(minutes=i*INTERVAL_MINUTES) for i in range(TOTAL_STEPS)]

        for i in range(TOTAL_STEPS):
            row = {
                "series_id": series_id,
                "DT": timestamps[i].strftime("%d.%m.%Y %H:%M"),
                "TPA": tpa_values[i] if not np.isnan(tpa_values[i]) else "",
                "Call Duration [min]": max(0, call_duration[i]),
                "Startcall Count": int(max(0, np.random.randn() * 2 + 1)),
                "Unanswered Call Count": int(max(0, np.random.randn() * 1)),
                "Revenue": max(0, revenue[i]),
                "Cost": max(0, cost[i]),
                "S_INTER_CNT": 0,
                "S_END_CNT": 0,
                "S_ALLOC_SUM": 0,
                "FK_DEST": "12345",
                "DEST": "CountryX",
                "is_fraud": is_fraud # Label for the whole series
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved generated data to {output_path}")
    print(df.head())
    print(f"Total rows: {len(df)}")
    print(f"Fraud cases: {df[df['is_fraud'] == True]['series_id'].nunique()}")

if __name__ == "__main__":
    generate_fraud_data()
