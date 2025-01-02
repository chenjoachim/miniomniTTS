import os
import glob
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from typing import Optional, Dict, Any

import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig

from model import *
from data import *

class SpeechUnitTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        optimizer_cls: torch.optim.Optimizer = AdamW,
        lr: float = 2e-4,
        lora_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        project_name: str = "speech_unit_training",
        checkpoint_dir: Optional[str] = None,
        **optimizer_kwargs
    ):
        if lora_config is not None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj"],
            }
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                bias=lora_config.get("bias", "none"),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        self.train_dataloader = train_dataset
        self.val_dataloader = None
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mimi_collate_fn, pin_memory=True
            )
        self.criterion = CrossEntropyLoss(ignore_index=0)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr, **optimizer_kwargs)

        if use_wandb:
            wandb.init(project=project_name)

        if self.checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_steps = 0
            progress_bar = tqdm(self.train_dataloader)
            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                loss = self._compute_batch_loss(input_ids, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                train_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({'train_loss': train_loss / train_steps})

                if self.use_wandb:
                    wandb.log({'train_loss': loss.item(), 'epoch': epoch})

            avg_train_loss = train_loss / train_steps

            if self.val_dataloader:
                val_loss = self.validate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pth")
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

            if self.checkpoint_dir and epoch % 3 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

    def _compute_batch_loss(self, input_ids, labels):
        # TODO: append begin-of-audio (2048) and end-of-audio (2049) index
        bos_labels = torch.full((8, 1), 2048).to(self.device)
        eos_labels = torch.full((8, 1), 2049).to(self.device)
        generate_audio_ids = torch.cat([bos_labels, labels], dim=-1)
        logits = self.model(input_ids=input_ids, audio_ids=generate_audio_ids)
        logits = logits.view(-1, 2050)
        
        add_tensor = torch.zeros_like(labels)
        for i in range(1, 8):
            add_tensor[i, :] = 2050 * (i)
        labels = labels - add_tensor
        labels = torch.cat([labels, eos_labels], dim=-1)
        return self.criterion(logits, labels.view(-1))

    def validate(self) -> float:
        self.model.eval()
        total_val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss = self._compute_batch_loss(input_ids, labels)
                total_val_loss += loss.item()
                val_steps += 1
        avg_val_loss = total_val_loss / val_steps
        if self.use_wandb:
            wandb.log({'val_loss': avg_val_loss})
        return avg_val_loss

    def save_checkpoint(self, filename: str):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        self._manage_checkpoints()

    def _manage_checkpoints(self, max_checkpoints=5):
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")))
        if len(checkpoint_files) > max_checkpoints:
            os.remove(checkpoint_files[0])


def main():
    from datasets import Dataset, load_dataset
    from itertools import islice
    from transformers import AutoTokenizer

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("anthony-wss/Soundon-tts", streaming="True")

    def collect_and_save(iterable_dataset, num_samples=256):
        collected_samples = list(islice(iterable_dataset, num_samples))
        return Dataset.from_list(collected_samples)

    test_ds = collect_and_save(ds['train'], num_samples=10000)
    base_model = AutoModelForCausalLM.from_pretrained( 
                model_name,
                device_map="cpu",
                torch_dtype="float",
                trust_remote_code=True,  
                # attn_implementation="flash_attention_2"
            )
    speech_model = SpeechUnitModel(base_model).to('cuda')

    lora_config = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj"]
    }

    train_dataset = MimiUnitDataset(test_ds, tokenizer)

    trainer = SpeechUnitTrainer(
        model=speech_model,
        train_dataset=train_dataset,
        val_dataset=None,
        batch_size=1,
        num_epochs=100,
        lr=1e-4,
        use_wandb=True,
        project_name="speech_unit_training",
        checkpoint_dir="checkpoints"
    )
    trainer.train()


if __name__ == "__main__":
    main()
