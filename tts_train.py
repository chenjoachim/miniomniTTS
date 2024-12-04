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
        if self.checkpoint_dir:
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
            if self.checkpoint_dir and epoch % 3 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step with Teacher Forcing."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Prepare for Teacher Forcing
        batch_size, seq_len = input_ids.shape
        loss = 0.0  # Accumulate loss over the sequence

        for t in range(1, seq_len-1):
            # Take the token at position `t` for all batches
            current_input_ids = input_ids[:, :t]  # Shape: (batch_size, seq_len)
            current_audio_ids = labels[:, :, :t]
            # current_attention_mask = attention_mask[:, t].unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Forward pass for the current token
            logits = self.model(input_ids=current_input_ids, audio_ids=current_audio_ids, attention_mask=attention_mask)
            # Shape: (batch_size, 8, codebook_size)
            
            # Calculate loss for the current token
            logits = logits.view(-1, 2050)
            current_labels = labels[:, :, t].view(-1)  # Shape: (batch_size,)
            loss += self.criterion(logits, current_labels)
            

        # Average loss over sequence length
        loss = loss / seq_len

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
        # Helper function to manage checkpoint files
        def manage_checkpoints(ckpt_dir, max_checkpoints=2):
            checkpoint_files = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth")))
            if len(checkpoint_files) > max_checkpoints:
                os.remove(checkpoint_files[0])  # Delete the oldest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        manage_checkpoints(self.checkpoint_dir)
    
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
    test_ds = collect_and_save(ds['train'], num_samples=100)

    from model import *
    from data import *

    # Initialize model and datasets
    config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    config.num_hidden_layers = 6
    speech_model = SpeechUnitModel(config)
    base_model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="cpu",
                torch_dtype="float",
                trust_remote_code=True,  
                # attn_implementation="flash_attention_2"
            )
    first_three_layers = {}
    for key, value in base_model.model.state_dict().items():
        if key.startswith("embed") or key.startswith("layers.0.") or key.startswith("layers.1.") or key.startswith("layers.2.") or key.startswith("layers.3.") or key.startswith("layers.4.") or key.startswith("layers.5.") or key.startswith("norm"):
            first_three_layers[key] = value
    speech_model.ref_model.load_state_dict(first_three_layers, strict=False)  # 'strict=False' allows partial loading
    speech_model = speech_model.to('cuda')

    train_dataset = MimiUnitDataset(test_ds, tokenizer)

    # val_dataset = MimiUnitDataset(...)
    trainer = SpeechUnitTrainer(
        model=speech_model,
        train_dataset=train_dataset,
        val_dataset=None,
        batch_size=32,
        num_epochs=300,
        lr=2e-4,
        use_wandb=False,  # Set to False if you don't want to use Weights & Biases
        project_name="speech_unit_training"
    )
    trainer.train()
