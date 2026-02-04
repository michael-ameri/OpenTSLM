import os
import pandas as pd
from datasets import Dataset
from typing import Tuple
import ast
from tqdm.auto import tqdm

# Constants
DATA_DIR = "data/fraud_detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

def parse_time_series(series_str):
    """
    Parse the time series string from the CSV into a list of floats.
    """
    try:
        if isinstance(series_str, list):
            return series_str
        return ast.literal_eval(series_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing time series: {e}")
        raise

def load_fraud_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess a Fraud Detection CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Parse the time series columns
    ts_columns = ['call_count', 'call_duration', 'revenue', 'cost']

    for col in ts_columns:
        print(f"Parsing {col}...")
        tqdm.pandas(desc=f"Parsing {col}")
        df[col] = df[col].progress_apply(parse_time_series)

    return df

def load_fraud_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the Fraud Detection dataset splits.

    Returns:
        Tuple of (train, validation, test) Dataset objects
    """
    train_df = load_fraud_csv(TRAIN_CSV)
    val_df = load_fraud_csv(VAL_CSV)
    test_df = load_fraud_csv(TEST_CSV)

    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset
