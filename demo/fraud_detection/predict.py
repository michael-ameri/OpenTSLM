
# demo/fraud_detection/predict.py

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from demo.fraud_detection.FraudDataset import FraudDataset

PATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "demo/fraud_detection/checkpoints/best_model.pt"
LLM_ID = "google/gemma-3-270m"


def predict():
    print(f"Using device: {DEVICE}")

    # 1. Load model architecture
    print(f"Loading model {LLM_ID}...")
    model = OpenTSLMFlamingo(
        cross_attn_every_n_layers=1,
        gradient_checkpointing=False,
        llm_id=LLM_ID,
        device=DEVICE
    ).to(DEVICE)
    model = model.float()

    # 2. Load trained weights
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 3. Load test data
    test_dataset = FraudDataset("test", EOS_TOKEN=model.get_eos_token())
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    )

    # 4. Run inference — call the underlying Flamingo generate without eos/pad token args
    print("Running predictions...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            input_ids, images, attention_mask, _ = model.pad_and_apply_batch(
                batch, include_labels=True
            )

            gen_ids = model.llm.generate(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
            )

            # Remove input tokens to get only the generated answer
            answer_ids = gen_ids[:, input_ids.shape[1]:]
            output = model.text_tokenizer.batch_decode(answer_ids, skip_special_tokens=True)

            print(f"\n--- Sample {i + 1} ---")
            print(f"Prediction: {output[0]}")


if __name__ == "__main__":
    predict()