import numpy as np
import json
import os
import random

# Constants
SEQ_LEN = 168  # 7 days * 24 hours
OUTPUT_DIR = "demo/fraud_detection/data"
TRAIN_SIZE = 100
TEST_SIZE = 20

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_diurnal_pattern(length=SEQ_LEN, base=10, amplitude=5):
    """Generates a daily cycle pattern."""
    x = np.arange(length)
    # 24-hour cycle (24 points)
    daily_cycle = np.sin(2 * np.pi * x / 24 - np.pi / 2)  # Peak around noon
    # Shift to positive range
    return base + amplitude * daily_cycle + np.random.normal(0, 1, length)

def normalize_series(series):
    """Z-score normalization."""
    mean = np.mean(series)
    std = np.std(series) + 1e-6
    return (series - mean) / std, mean, std

def generate_sample(sample_id):
    is_fraud = random.choice([True, False])

    # Base Patterns (Normal Behavior)
    # Duration: High during day, low at night
    duration = generate_diurnal_pattern(base=500, amplitude=300)
    duration = np.maximum(duration, 0) # Non-negative

    # Count: Follows duration somewhat
    count = generate_diurnal_pattern(base=50, amplitude=30)
    count = np.maximum(count, 0)

    # Revenue: Correlated with duration
    revenue = duration * 0.5 + np.random.normal(0, 50, SEQ_LEN)
    revenue = np.maximum(revenue, 0)

    tpa_description = "No Fraud Flag"
    answer_reasoning = ""
    verdict = "Normal"

    if is_fraud:
        fraud_type = random.choice(["Wangiri", "IRSF"])

        # Inject Anomaly
        start_idx = random.randint(20, SEQ_LEN - 10)
        end_idx = start_idx + random.randint(2, 6)

        if fraud_type == "Wangiri":
            # High Count, Low Duration
            spike_magnitude = random.randint(200, 500)
            count[start_idx:end_idx] += spike_magnitude
            duration[start_idx:end_idx] = np.random.uniform(0, 2, end_idx - start_idx) # Very short calls

            tpa_description = "High velocity of short duration calls detected."
            answer_reasoning = (
                f"Analysis of Stream 2 (Startcall Count): A significant spike is observed around index {start_idx}, "
                f"reaching values roughly {spike_magnitude/np.mean(count):.1f}x the baseline.\n"
                f"Analysis of Stream 1 (Call Duration): Simultaneously, the duration drops to near zero.\n"
                f"Analysis of Stream 3 (Revenue): Revenue remains low but non-zero.\n"
                "Synthesis: This pattern of high velocity with negligible duration is characteristic of Wangiri fraud."
            )
            verdict = "Fraud"

        elif fraud_type == "IRSF":
            # High Revenue, High Duration (or sometimes just high revenue per min)
            revenue_spike = random.randint(5000, 20000)
            revenue[start_idx:end_idx] += revenue_spike
            duration[start_idx:end_idx] += random.randint(100, 300) # Increased usage

            tpa_description = "Abnormal revenue spike detected."
            answer_reasoning = (
                f"Analysis of Stream 3 (Revenue): A massive spike is observed around index {start_idx}, "
                f"inconsistent with historical usage.\n"
                f"Analysis of Stream 1 (Call Duration): Duration also increases but disproportionately less than revenue.\n"
                "Synthesis: The extreme revenue generation suggests International Revenue Share Fraud (IRSF)."
            )
            verdict = "Fraud"
    else:
        answer_reasoning = (
            "All streams exhibit standard diurnal seasonality with no significant outliers consistent with fraud patterns. "
            "Revenue and duration are well-correlated."
        )
        verdict = "Normal"

    # Preprocessing
    # Log transform Revenue
    revenue_log = np.log1p(revenue)

    # Normalize
    dur_norm, dur_mean, dur_std = normalize_series(duration)
    cnt_norm, cnt_mean, cnt_std = normalize_series(count)
    rev_norm, rev_mean, rev_std = normalize_series(revenue_log)

    final_answer = f"{answer_reasoning}\nAnswer: {verdict}"

    return {
        "id": f"sample_{sample_id}",
        "time_series": [dur_norm.tolist(), cnt_norm.tolist(), rev_norm.tolist()],
        "stats": [
            {"name": "Duration", "mean": float(dur_mean), "std": float(dur_std)},
            {"name": "Count", "mean": float(cnt_mean), "std": float(cnt_std)},
            {"name": "Revenue Log", "mean": float(rev_mean), "std": float(rev_std)}
        ],
        "tpa_description": tpa_description,
        "answer": final_answer
    }

def generate_dataset(num_samples, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        for i in range(num_samples):
            sample = generate_sample(i)
            f.write(json.dumps(sample) + '\n')
    print(f"Generated {num_samples} samples in {filepath}")

if __name__ == "__main__":
    np.random.seed(42)
    generate_dataset(TRAIN_SIZE, "train.jsonl")
    generate_dataset(TEST_SIZE, "test.jsonl")
