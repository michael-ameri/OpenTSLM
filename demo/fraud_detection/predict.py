
# demo/fraud_detection/predict.py

import os
import sys
import re
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
MAX_NEW_TOKENS = 150


def extract_label(text: str) -> str:
    """Extract the classification label (Fraud or Normal) from model output."""
    text_lower = text.lower()
    # Look for explicit "answer:" pattern first
    answer_match = re.search(r'answer\s*:\s*(fraud|normal)', text_lower)
    if answer_match:
        return answer_match.group(1).capitalize()
    # Fall back to keyword detection
    if "spike" in text_lower or "inconsistent" in text_lower or "fraud" in text_lower:
        return "Fraud"
    if "no significant outliers" in text_lower or "normal" in text_lower or "well-correlated" in text_lower:
        return "Normal"
    return "Unknown"


def extract_reasoning(text: str) -> str:
    """Extract the first complete sentence as a concise reasoning summary."""
    text = text.strip()
    # Remove duplicated "Answer: XAnswer:" artifacts
    text = re.sub(r'Answer\s*:\s*\w+Answer\s*:', '', text).strip()
    # Get the first sentence
    match = re.match(r'(.+?[.!])', text)
    if match:
        return match.group(1).strip()
    # If no sentence-ending punctuation, return up to first newline or full text
    first_line = text.split('\n')[0].strip()
    return first_line


def get_ground_truth(sample: dict) -> str:
    """Extract ground truth label from the sample's answer field."""
    answer = sample.get("answer", "")
    # The answer field contains full reasoning ending with "Answer: Fraud" or "Answer: Normal"
    match = re.search(r'Answer\s*:\s*(Fraud|Normal)', answer, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # Fallback: check for keywords anywhere
    answer_lower = answer.lower()
    if "fraud" in answer_lower:
        return "Fraud"
    if "normal" in answer_lower:
        return "Normal"
    return "Unknown"


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

    # 4. Run inference with accuracy tracking
    print(f"Running predictions (max_new_tokens={MAX_NEW_TOKENS})...\n")

    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            input_ids, images, attention_mask, _ = model.pad_and_apply_batch(
                batch, include_labels=True
            )

            gen_ids = model.llm.generate(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            # Remove input tokens to get only the generated answer
            answer_ids = gen_ids[:, input_ids.shape[1]:]
            raw_output = model.text_tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]

            # Post-process
            predicted_label = extract_label(raw_output)
            reasoning = extract_reasoning(raw_output)
            ground_truth = get_ground_truth(batch[0])

            is_correct = predicted_label == ground_truth
            if predicted_label != "Unknown" and ground_truth != "Unknown":
                total += 1
                if is_correct:
                    correct += 1

            status = "✅" if is_correct else "❌"
            results.append((i + 1, predicted_label, ground_truth, is_correct, reasoning))

            print(f"\n--- Sample {i + 1} ---")
            print(f"  Predicted: {predicted_label}  |  Ground Truth: {ground_truth}  {status}")
            print(f"  Reasoning: {reasoning}")

    # 5. Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)\n")

    # Confusion matrix counts
    tp = sum(1 for _, p, g, _, _ in results if p == "Fraud" and g == "Fraud")
    tn = sum(1 for _, p, g, _, _ in results if p == "Normal" and g == "Normal")
    fp = sum(1 for _, p, g, _, _ in results if p == "Fraud" and g == "Normal")
    fn = sum(1 for _, p, g, _, _ in results if p == "Normal" and g == "Fraud")

    print(f"  True Positives  (Fraud → Fraud):   {tp}")
    print(f"  True Negatives  (Normal → Normal): {tn}")
    print(f"  False Positives (Normal → Fraud):  {fp}")
    print(f"  False Negatives (Fraud → Normal):  {fn}")

    if tp + fp > 0:
        precision = tp / (tp + fp) * 100
        print(f"\n  Precision: {precision:.1f}%")
    if tp + fn > 0:
        recall = tp / (tp + fn) * 100
        print(f"  Recall:    {recall:.1f}%")

    # List misclassified samples
    misclassified = [r for r in results if not r[3]]
    if misclassified:
        print(f"\nMisclassified samples:")
        for sample_num, pred, gt, _, reasoning in misclassified:
            print(f"  Sample {sample_num}: predicted {pred}, actual {gt}")
    else:
        print(f"\n🎉 Perfect classification on all {total} samples!")


if __name__ == "__main__":
    predict()