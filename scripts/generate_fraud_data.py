import os
import pandas as pd
import numpy as np
import random

# Constants
NUM_SAMPLES = 500
TIME_STEPS = 96  # 24 hours * 4 (15 min buckets)
DATA_DIR = "data/fraud_detection"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_time_series(is_fraud=False):
    # Base pattern (daily cycle)
    x = np.linspace(0, 4*np.pi, TIME_STEPS)
    base = np.sin(x) + 2  # Shift up to be positive

    # Noise
    noise = np.random.normal(0, 0.2, TIME_STEPS)

    # Generate features
    if is_fraud:
        # Fraud pattern: sudden spike
        spike_start = random.randint(20, 70)
        spike_end = spike_start + random.randint(5, 15)
        spike_magnitude = random.uniform(3, 6)

        call_count = base + noise
        call_count[spike_start:spike_end] += spike_magnitude

        call_duration = base * random.uniform(0.8, 1.2) + noise
        call_duration[spike_start:spike_end] += spike_magnitude * random.uniform(0.5, 1.5)

        revenue = call_duration * random.uniform(0.1, 0.2)
        cost = call_duration * random.uniform(0.05, 0.1) # Cost is usually lower

        # In fraud, maybe cost spikes disproportionately or revenue is artificial
        if random.random() > 0.5:
            cost[spike_start:spike_end] *= 2.0 # High cost fraud (e.g. premium rate numbers)

        rationale = f"Fraud detected due to abnormal spike in call activity between time steps {spike_start} and {spike_end}."

    else:
        call_count = base + noise
        call_duration = base * random.uniform(0.8, 1.2) + noise
        revenue = call_duration * random.uniform(0.1, 0.2)
        cost = call_duration * random.uniform(0.05, 0.1)

        rationale = "Normal activity pattern observed. No irregularities detected."

    # Normalize to avoid negative values
    call_count = np.maximum(call_count, 0)
    call_duration = np.maximum(call_duration, 0)
    revenue = np.maximum(revenue, 0)
    cost = np.maximum(cost, 0)

    return {
        "call_count": call_count.tolist(),
        "call_duration": call_duration.tolist(),
        "revenue": revenue.tolist(),
        "cost": cost.tolist(),
        "label": "Fraud" if is_fraud else "Normal",
        "rationale": rationale
    }

def generate_dataset():
    data = []
    for _ in range(NUM_SAMPLES):
        is_fraud = random.random() < 0.3 # 30% fraud rate
        sample = generate_time_series(is_fraud)
        data.append(sample)

    df = pd.DataFrame(data)

    # Split
    train_size = int(0.7 * NUM_SAMPLES)
    val_size = int(0.15 * NUM_SAMPLES)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    # Save
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    print(f"Generated {NUM_SAMPLES} samples.")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Saved to {DATA_DIR}")

if __name__ == "__main__":
    generate_dataset()
