
import os
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from datasets import Dataset

class FraudDetectionDataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        # Load the generated CSV
        csv_path = "demo/fraud_detection/fraud_data.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found at {csv_path}. Run generate_data.py first.")

        df = pd.read_csv(csv_path)

        # Group by series_id to get episodes
        grouped = df.groupby("series_id")
        episodes = []
        for _, group in grouped:
            episodes.append(group)

        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(episodes)

        n_total = len(episodes)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)

        train_episodes = episodes[:n_train]
        val_episodes = episodes[n_train:n_train+n_val]
        test_episodes = episodes[n_train+n_val:]

        # Helper to convert list of DataFrames to HuggingFace Dataset
        def to_hf_dataset(episode_list):
            data_dict = {"series_id": [], "data": [], "is_fraud": [], "tpa_triggered": []}
            for episode in episode_list:
                data_dict["series_id"].append(episode["series_id"].iloc[0])
                # Store the dataframe as a dictionary record list or similar
                # We can just store the columns we need
                episode_data = episode[["Call Duration [min]", "Startcall Count",
                                      "Unanswered Call Count", "Revenue", "Cost", "TPA"]].to_dict(orient="list")
                data_dict["data"].append(episode_data)
                data_dict["is_fraud"].append(bool(episode["is_fraud"].iloc[0]))

                # Check if TPA triggered
                tpa_values = episode["TPA"].fillna("").astype(str)
                triggered = any(v != "" and v != "nan" for v in tpa_values)
                data_dict["tpa_triggered"].append(triggered)

            return Dataset.from_dict(data_dict)

        return to_hf_dataset(train_episodes), to_hf_dataset(val_episodes), to_hf_dataset(test_episodes)

    def _get_answer(self, row) -> str:
        return "Yes" if row["is_fraud"] else "No"

    def _get_pre_prompt(self, row) -> str:
        prompt = "Analyze the following mobile carrier data for fraud."
        if row["tpa_triggered"]:
            prompt += " A TPA rule was triggered during this period."
        else:
            prompt += " No TPA rules were triggered."
        return prompt

    def _get_post_prompt(self, row) -> str:
        return "Based on the time series data and TPA status, is this considered actual fraud? Answer Yes or No."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        data = row["data"]
        prompts = []

        # Define features to include
        features = [
            ("Revenue", "Revenue"),
            ("Cost", "Cost"),
            ("Call Duration [min]", "Call Duration"),
            ("Startcall Count", "Start Call Count"),
            ("Unanswered Call Count", "Unanswered Call Count")
        ]

        for col, name in features:
            series = np.array(data[col], dtype=np.float32)
            # Handle NaNs if any (replace with 0)
            series = np.nan_to_num(series)

            # Simple stats for the prompt text
            mean_val = np.mean(series)
            std_val = np.std(series)

            prompt_text = f"This is the {name} time series (mean={mean_val:.2f}, std={std_val:.2f})."
            prompts.append(TextTimeSeriesPrompt(prompt_text, series))

        return prompts
