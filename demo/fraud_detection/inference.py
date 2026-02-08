
import os
import argparse
import torch
from torch.utils.data import DataLoader
import sys
import json

# Add src to path
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("demo/fraud_detection"))

from FraudDetectionDataset import FraudDetectionDataset
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Model
    print(f"Initializing model {args.model_id}...")
    try:
        model = OpenTSLMFlamingo(
            llm_id=args.model_id,
            device=device,
            cross_attn_every_n_layers=1
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Checkpoint
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle state dict structure
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint # Assume direct state dict

        # Remove "module." prefix if it exists (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Warning loading checkpoint: {e}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using base model.")

    model.eval()

    # Load Test Dataset
    print("Loading test dataset...")
    test_dataset = FraudDetectionDataset("test", EOS_TOKEN=model.get_eos_token())

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)
    )

    print(f"Running inference on {args.num_samples} samples...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.num_samples:
                break

            predictions = model.generate(batch, max_new_tokens=200)

            for sample, pred in zip(batch, predictions):
                print("-" * 50)
                print(f"Sample {i+1}")
                print(f"Pre-prompt: {sample['pre_prompt']}")
                print(f"Features: {len(sample['time_series_text'])} time series prompts")
                print(f"Question: {sample['post_prompt']}")
                print(f"Gold Answer: {sample['answer']}")
                print(f"Model Prediction: {pred}")
                print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B", help="HuggingFace model ID")
    parser.add_argument("--checkpoint", type=str, default="demo/fraud_detection/checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    args = parser.parse_args()

    run_inference(args)
