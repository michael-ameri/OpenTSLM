from datasets import Dataset
from typing import List, Tuple, Literal
import torch
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.fraud_detection.fraud_detection_loader import load_fraud_splits

TIME_SERIES_LABELS = [
    "The following is the call count data",
    "The following is the call duration data",
    "The following is the revenue data",
    "The following is the cost data",
]

class FraudDetectionDataset(QADataset):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str, format_sample_str: bool = False, time_series_format_function=None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_fraud_splits()

    def _get_answer(self, row) -> str:
        return row["rationale"]

    def _get_pre_prompt(self, _row) -> str:
        text = """
        You are a fraud detection analyst for a mobile carrier. You are given time series data for call count, call duration, revenue, and cost in 15-minute buckets.

        Your task is to analyze these time series for any abnormal patterns that might indicate fraud.

        Instructions:
        - Analyze the patterns in call volume and duration.
        - Check for sudden spikes or irregularities in revenue and cost.
        - Determine if the activity is "Fraud" or "Normal".
        - Provide a rationale for your decision.
        - End your response with the final decision.
        """
        return text

    def _get_post_prompt(self, _row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # Extract the time series data from the row
        # Order: call_count, call_duration, revenue, cost
        series = torch.tensor(
            [
                row["call_count"],
                row["call_duration"],
                row["revenue"],
                row["cost"],
            ],
            dtype=torch.float32,
        )

        # Check for invalid data
        if torch.isnan(series).any() or torch.isinf(series).any():
             raise ValueError("Invalid data detected")

        # Normalize the data
        means = series.mean(dim=1, keepdim=True)
        stds = series.std(dim=1, keepdim=True)

        # Handle zero or very small standard deviations
        min_std = 1e-6
        stds = torch.clamp(stds, min=min_std)

        series_norm = (series - means) / stds

        prompts = []
        for i, (time_series_label, time_series, mean, std) in enumerate(zip(
            TIME_SERIES_LABELS,
            series_norm.tolist(),
            means.squeeze().tolist(),
            stds.squeeze().tolist()
        )):
            text_prompt = f"{time_series_label}, it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, time_series))
        return prompts

if __name__ == "__main__":
    dataset = FraudDetectionDataset(split="train", EOS_TOKEN="")
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        print("\nSample data:")
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Sample pre prompt:", sample["pre_prompt"][:200] + "...")
        print("Sample answer:", sample["answer"])
