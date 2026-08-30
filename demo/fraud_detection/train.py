
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import sys
import traceback

# Add src to path
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("demo/fraud_detection"))

from FraudDetectionDataset import FraudDetectionDataset
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, BATCH_SIZE

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Model
    print(f"Initializing model {args.model_id}...")
    try:
        model = OpenTSLMFlamingo(
            llm_id=args.model_id,
            device=device,
            cross_attn_every_n_layers=1, # Default from CurriculumTrainer
            gradient_checkpointing=args.gradient_checkpointing
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        print("Make sure you have access to the model on HuggingFace and have logged in via `hf auth login`.")
        return

    # Initialize Dataset
    print("Loading datasets...")
    train_dataset = FraudDetectionDataset("train", EOS_TOKEN=model.get_eos_token())
    val_dataset = FraudDetectionDataset("validation", EOS_TOKEN=model.get_eos_token())

    # DataLoaders
    collate_fn = lambda batch: extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # Validate one by one
        shuffle=False,
        collate_fn=collate_fn
    )

    # Optimizer
    # Simplify optimizer setup (fine-tune all trainable params)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training Loop
    best_val_loss = float("inf")
    save_dir = "demo/fraud_detection/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                val_loss += model.compute_loss(batch).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, "best_model.pt")

            # Save relevant state
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss
            }, save_path)
            print(f"New best model saved to {save_path}")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B", help="HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    train(args)
