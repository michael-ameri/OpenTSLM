# SPDX-License-Identifier: MIT

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

# Add src and current directory to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from demo.fraud_detection.FraudDataset import FraudDataset

# Default Config
BATCH_SIZE = 2
PATCH_SIZE = 4
NUM_EPOCHS = 3
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_ID_DEFAULT = "google/gemma-3-270m" # Smaller model for demo/testing

def train(args):
    print(f"Using device: {DEVICE}")

    # 1. Load Model
    print(f"Loading model {args.llm_id}...")
    try:
        model = OpenTSLMFlamingo(
            cross_attn_every_n_layers=1,
            gradient_checkpointing=False,
            llm_id=args.llm_id,
            device=DEVICE
        ).to(DEVICE)
        # todo mike: run locally
        model = model.float()
    except Exception as e:
        print(f"Failed to load model {args.llm_id}: {e}")
        print("Ensure you have access to the model or use a different --llm_id.")
        return

    # 2. Load Dataset
    print("Loading datasets...")
    train_dataset = FraudDataset("train", EOS_TOKEN=model.get_eos_token())
    test_dataset = FraudDataset("test", EOS_TOKEN=model.get_eos_token())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    )

    # 3. Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                val_loss += model.compute_loss(batch).item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch} - Avg Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint if best
        checkpoint_dir = "demo/fraud_detection/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

        # Save epoch checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection Model")
    parser.add_argument("--llm_id", type=str, default=LLM_ID_DEFAULT, help="HuggingFace Model ID")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")

    args = parser.parse_args()
    train(args)
