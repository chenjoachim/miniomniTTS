import os
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from typing import Optional, Dict, Any

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
        batch_size: int = 8,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        project_name: str = "speech_unit_training",
        checkpoint_dir: Optional[str] = None,
        **optimizer_kwargs
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mimi_collate_fn
        )
        
        self.val_dataloader = None
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=mimi_collate_fn
            )
        
        # Initialize optimizer
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=lr,
            **optimizer_kwargs
        )
        
        # Initialize loss function
        self.criterion = CrossEntropyLoss(ignore_index=0)
        
        # Initialize wandb
        if use_wandb:
            wandb.init(project=project_name)
            
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                loss = self.training_step(batch)
                
                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * self.gradient_accumulation_steps
                train_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'train_loss': train_loss / train_steps
                })
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'epoch': epoch
                    })
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            if self.val_dataloader:
                val_loss = self.validate()
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pth")
                
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")
            
            # Save checkpoint
            if self.checkpoint_dir:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step"""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        # Reshape outputs and labels for loss calculation
        # outputs shape: (batch_size, num_heads, seq_len, codebook_size)
        # labels shape: (batch_size, num_heads, seq_len)
        batch_size, num_heads, seq_len, codebook_size = outputs.shape
        outputs = outputs.view(-1, codebook_size)  # Flatten for CrossEntropyLoss
        labels = labels.view(-1)  # Flatten labels
        
        loss = self.criterion(outputs, labels)
        
        return loss
    
    def validate(self) -> float:
        """Perform validation and return validation loss"""
        self.model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                loss = self.training_step(batch)
                total_val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps
        
        if self.use_wandb:
            wandb.log({'val_loss': avg_val_loss})
        
        return avg_val_loss
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    from datasets import Dataset, load_dataset
    from itertools import islice
    from transformers import AutoTokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("anthony-wss/Soundon-tts", streaming="True")
    def collect_and_save(iterable_dataset, num_samples=256):
        """
        Collects a specified number of samples from an IterableDataset and saves them as a Hugging Face Dataset.

        Args:
            iterable_dataset (IterableDataset): Input IterableDataset to sample from.
            output_path (str): Path to save the Hugging Face Dataset.
            num_samples (int): Number of samples to collect.
        """
        # Collect `num_samples` items from the iterable dataset
        collected_samples = list(islice(iterable_dataset, num_samples))

        # Convert to a Hugging Face Dataset
        hf_dataset = Dataset.from_list(collected_samples)
        return hf_dataset
    test_ds = collect_and_save(ds['train'], num_samples=64)

    from model import *
    from data import *

    # Initialize your model and datasets
    model = SpeechUnitModel()

    train_dataset = MimiUnitDataset(test_ds, tokenizer)

    # val_dataset = MimiUnitDataset(...)
    trainer = SpeechUnitTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        batch_size=16,
        num_epochs=3,
        lr=2e-4,
        use_wandb=False,  # Set to False if you don't want to use Weights & Biases
        # project_name="speech_unit_training"
    )
    trainer.train()
