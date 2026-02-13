# SPDX-License-Identifier: MIT

import json
import os
from typing import List, Tuple, Literal
from torch.utils.data import Dataset
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
import numpy as np

class FraudDataset(QADataset):
    """
    Dataset for Fraud Detection in Mobile Carrier Time Series.
    """

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Loads train and test splits.
        Note: validation will be a subset of train or same as test for this demo.
        """
        train_path = os.path.join(self.DATA_DIR, "train.jsonl")
        test_path = os.path.join(self.DATA_DIR, "test.jsonl")

        with open(train_path, "r") as f:
            train_data = [json.loads(line) for line in f]

        with open(test_path, "r") as f:
            test_data = [json.loads(line) for line in f]

        # For this demo, we use test as validation
        return train_data, test_data, test_data

    def _get_answer(self, row) -> str:
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        return "Analyze the following 7-day CDR data."

    def _get_post_prompt(self, row) -> str:
        return f"TPA Status: {row['tpa_description']}. Assess for fraud."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Constructs the list of TextTimeSeriesPrompt objects, one for each channel.
        """
        prompts = []
        channels = row["time_series"] # List of 3 lists
        stats = row["stats"] # List of 3 dicts

        # Channel 0: Duration
        dur_stats = stats[0]
        dur_text = f"Stream 1 ({dur_stats['name']}): Mean={dur_stats['mean']:.2f}, Std={dur_stats['std']:.2f}."
        prompts.append(TextTimeSeriesPrompt(dur_text, channels[0]))

        # Channel 1: Count
        cnt_stats = stats[1]
        cnt_text = f"Stream 2 ({cnt_stats['name']}): Mean={cnt_stats['mean']:.2f}, Std={cnt_stats['std']:.2f}."
        prompts.append(TextTimeSeriesPrompt(cnt_text, channels[1]))

        # Channel 2: Revenue
        rev_stats = stats[2]
        rev_text = f"Stream 3 ({rev_stats['name']}): Mean={rev_stats['mean']:.2f}, Std={rev_stats['std']:.2f}."
        prompts.append(TextTimeSeriesPrompt(rev_text, channels[2]))

        return prompts
