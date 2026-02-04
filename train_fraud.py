
import os
import json
import argparse
from opentslm.time_series_datasets.fraud_detection.FraudDetectionDataset import FraudDetectionDataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import (
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)

# Define our single stage
FRAUD_STAGE = "stage_fraud_detection"

def collate_fn(batch):
    return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)

class FraudTrainer:
    def __init__(
        self,
        model_type: str,
        device: str = None,
        gradient_checkpointing: bool = False,
        llm_id: str = None,
    ):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_id = llm_id
        self.llm_id_safe = self._sanitize_llm_id(llm_id)
        self.gradient_checkpointing = gradient_checkpointing

        # Simplified: No distributed training support for this script
        self.rank = 0
        self.world_size = 1

        self.model = self._initialize_model()
        self.results_dir = os.path.join("results", self.llm_id_safe, self.model_type)
        self._create_results_dir()

    def _sanitize_llm_id(self, llm_id: str) -> str:
        if not llm_id:
            return "unknown_llm"
        return llm_id.split("/")[-1].replace(".", "_").replace("-", "_")

    def _initialize_model(self):
        print(f"Initializing {self.model_type} on {self.device}...")
        if self.model_type == "OpenTSLMSP":
            model = OpenTSLMSP(llm_id=self.llm_id, device=self.device).to(self.device)
        elif self.model_type == "OpenTSLMFlamingo":
            model = OpenTSLMFlamingo(
                cross_attn_every_n_layers=1,
                gradient_checkpointing=self.gradient_checkpointing,
                llm_id=self.llm_id,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return model

    def _create_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)
        stage_dir = os.path.join(self.results_dir, FRAUD_STAGE)
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "results"), exist_ok=True)

    def _get_optimizer(self, lr_encoder=None, lr_projector=None, lr_base=None):
        if self.model_type == "OpenTSLMSP":
            enc_params = list(self.model.encoder.parameters())
            proj_params = list(self.model.projector.projector.parameters())

            encoder_lr = lr_encoder or LR_ENCODER
            projector_lr = lr_projector or LR_PROJECTOR

            param_groups = [
                {"params": enc_params, "lr": encoder_lr, "weight_decay": WEIGHT_DECAY},
                {"params": proj_params, "lr": projector_lr, "weight_decay": WEIGHT_DECAY},
            ]

            # Add LoRA params if enabled
            if getattr(self.model, "lora_enabled", False):
                lora_params = self.model.get_lora_parameters()
                param_groups.append({
                    "params": lora_params,
                    "lr": projector_lr,
                    "weight_decay": WEIGHT_DECAY,
                })

            return AdamW(param_groups)
        else:
            # Flamingo
            base_lr = lr_base or 2e-4
            return AdamW(self.model.parameters(), lr=base_lr, weight_decay=WEIGHT_DECAY)

    def _save_checkpoint(self, epoch, val_loss, optimizer, scheduler):
        checkpoint_dir = os.path.join(self.results_dir, FRAUD_STAGE, "checkpoints")
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }

        if self.model_type == "OpenTSLMSP":
            checkpoint["encoder_state"] = self.model.encoder.state_dict()
            checkpoint["projector_state"] = self.model.projector.state_dict()
            self.model.save_lora_state_to_checkpoint(checkpoint)
        else:
            checkpoint["model_state"] = self.model.state_dict()

        torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
        print(f"Saved best model to {checkpoint_dir}")

    def train(self, num_epochs=10, batch_size=4):
        print(f"Starting training for {FRAUD_STAGE}")

        # Dataset
        train_dataset = FraudDetectionDataset("train", EOS_TOKEN=self.model.get_eos_token())
        val_dataset = FraudDetectionDataset("validation", EOS_TOKEN=self.model.get_eos_token())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Optimizer & Scheduler
        optimizer = self._get_optimizer()
        total_steps = num_epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRAC * total_steps), total_steps)

        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for batch in pbar:
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = running_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    val_loss += self.model.compute_loss(batch).item()
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_checkpoint(epoch, avg_val_loss, optimizer, scheduler)

    def evaluate(self):
        print(f"Evaluating {FRAUD_STAGE}")
        test_dataset = FraudDetectionDataset("test", EOS_TOKEN=self.model.get_eos_token())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Load best model
        checkpoint_path = os.path.join(self.results_dir, FRAUD_STAGE, "checkpoints", "best_model.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading best model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if self.model_type == "OpenTSLMSP":
                self.model.encoder.load_state_dict(checkpoint["encoder_state"])
                self.model.projector.load_state_dict(checkpoint["projector_state"])
                self.model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
            else:
                self.model.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            print("No checkpoint found, evaluating with current weights.")

        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                predictions = self.model.generate(batch, max_new_tokens=200)
                for sample, pred in zip(batch, predictions):
                    results.append({
                        "generated": pred,
                        "gold": sample["answer"]
                    })
                    print(f"Gold: {sample['answer']}")
                    print(f"Pred: {pred}")
                    print("-" * 20)

        # Save results
        results_file = os.path.join(self.results_dir, FRAUD_STAGE, "results", "predictions.jsonl")
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved results to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["OpenTSLMSP", "OpenTSLMFlamingo"])
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()

    trainer = FraudTrainer(args.model, llm_id=args.llm_id)

    if not args.eval_only:
        trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)

    trainer.evaluate()
