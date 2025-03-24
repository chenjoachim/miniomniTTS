import os
import glob
import wandb
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from typing import Optional, Dict, Any

# import peft
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import argparse

from model import SpeechUnitModel
from data import mimi_collate_fn, MimiUnitDataset

def setup_ddp():
    """
    Initialize the distributed environment when using torchrun.
    torchrun sets up the necessary environment variables automatically.
    """
    # With torchrun, environment variables are already set
    dist.init_process_group("nccl")

class SpeechUnitTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        optimizer_cls: torch.optim.Optimizer = AdamW,
        lr: float = 2e-4,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        lora_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        project_name: str = "speech_unit_training",
        checkpoint_dir: Optional[str] = None,
        codebook_size: int = 2048, # NOT Including BOS and EOS tokens
        vocoder_layer: int = 8,
        rank: int = -1,
        world_size: int = 1,
        **optimizer_kwargs
    ):

        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        self.codebook_size = codebook_size
        self.vocoder_layer = vocoder_layer
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        # Set up DDP if applicable
        if self.is_distributed:
            # Move model to the correct device
            self.model = self.model.to(self.rank)
            # Wrap model with DDP
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)
            
            # Create distributed samplers
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            
            self.train_dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                sampler=train_sampler,
                collate_fn=mimi_collate_fn, 
                pin_memory=True
            )
            
            if val_dataset:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )
                self.val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    sampler=val_sampler,
                    collate_fn=mimi_collate_fn, 
                    pin_memory=True
                )
            else:
                self.val_dataloader = None
        else:
            # Regular non-distributed data loading
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, 
                collate_fn=mimi_collate_fn, pin_memory=True
            )
            self.val_dataloader = None
            if val_dataset:
                self.val_dataloader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, 
                    collate_fn=mimi_collate_fn, pin_memory=True
                )

        self.criterion = CrossEntropyLoss(ignore_index=0)
        
        # Initialize optimizer - in DDP mode, we only need to optimize parameters of the wrapped model
        self.optimizer = optimizer_cls(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            **optimizer_kwargs
        )
        
        # Initialize scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(warmup_ratio * self.num_epochs * len(self.train_dataloader)), 
            num_training_steps=self.num_epochs * len(self.train_dataloader)
        )
        
        self.val_interval = 1000
        
        # Only initialize wandb on the main process
        if use_wandb and (not self.is_distributed or self.rank == 0):
            wandb.init(project=project_name)

        if self.checkpoint_dir and (not self.is_distributed or self.rank == 0):
            os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            # Set epoch for DistributedSampler
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
                
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            # Only show progress bar on main process
            if not self.is_distributed or self.rank == 0:
                progress_bar = tqdm(self.train_dataloader)
            else:
                progress_bar = self.train_dataloader
                
            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                loss = self._compute_batch_loss(input_ids, labels)
                loss.backward()
                
                # Calculate gradient norm before clipping
                grad_norm = self._get_grad_norm()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Calculate gradient norm after clipping
                clipped_grad_norm = self._get_grad_norm()
                
                self.optimizer.step()
                self.scheduler.step()

                train_loss += loss.item()
                train_steps += 1
                
                # Update progress bar on main process only
                if not self.is_distributed or self.rank == 0:
                    progress_bar.set_postfix({
                        'train_loss': train_loss / train_steps,
                        'grad_norm': grad_norm,
                        'clipped_grad_norm': clipped_grad_norm
                    })
                
                # Log to wandb from main process only
                if self.use_wandb and (not self.is_distributed or self.rank == 0):
                    wandb.log({
                        'train_loss': loss.item(),
                        'lr': self.scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'grad_norm': grad_norm,
                        'clipped_grad_norm': clipped_grad_norm,
                        'grad_norm_ratio': clipped_grad_norm / grad_norm if grad_norm > 0 else 0
                    })
                    
                # Validate on intervals (only from main process if distributed)
                if self.val_dataloader and step % self.val_interval == 0 and step > 0 and (not self.is_distributed or self.rank == 0):
                    val_loss = self.validate()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(f"best_model.pth")
                    print(f"Epoch {epoch + 1}, Step {step}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")

            avg_train_loss = train_loss / train_steps
            
            # Only run validation on main process if distributed
            if self.val_dataloader and (not self.is_distributed or self.rank == 0):
                val_loss = self.validate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pth")
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            elif not self.is_distributed or self.rank == 0:
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

            # Save checkpoint periodically (only from main process if distributed)
            if self.checkpoint_dir and (epoch+1) % 20 == 0 and (not self.is_distributed or self.rank == 0):
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
                
            # Make sure all processes are synchronized at the end of each epoch
            if self.is_distributed:
                dist.barrier()

    def _get_grad_norm(self) -> float:
        """Calculate the gradient norm for all parameters that require gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _compute_batch_loss(self, input_ids, labels):
        batch_size = input_ids.size(0)
        assert batch_size == labels.size(0), f"Input and labels must have the same batch size. Got input {batch_size} and label {labels.size(0)}"
        bos_labels = torch.full((batch_size, self.vocoder_layer, 1), self.codebook_size).to(self.device)
        eos_labels = torch.full((batch_size, self.vocoder_layer, 1), self.codebook_size+1).to(self.device)
        generate_audio_ids = torch.cat([bos_labels, labels], dim=-1)
        logits = self.model(input_ids=input_ids, audio_ids=generate_audio_ids)
        logits = logits.view(-1, self.codebook_size+2)
        # print("logits shape: ", logits.shape)
        # print("labels shape: ", labels.shape)
        add_tensor = torch.zeros_like(labels)
        for i in range(1, self.vocoder_layer):
            add_tensor[:, i, :] = self.codebook_size * (i)
        labels = labels - add_tensor
        labels = torch.cat([labels, eos_labels], dim=-1)
        return self.criterion(logits, labels.view(-1))

    def validate(self) -> float:
        """Validation should only be called from the main process in DDP mode."""
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
        if self.use_wandb and (not self.is_distributed or self.rank == 0):
            wandb.log({'val_loss': avg_val_loss})
        return avg_val_loss

    def save_checkpoint(self, filename: str):
        """Only save checkpoint from main process."""
        if self.is_distributed:
            # In DDP mode, only save from rank 0
            if self.rank == 0:
                # Save the underlying model (not the DDP wrapper)
                checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                torch.save(self.model.module.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                self._manage_checkpoints()
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            self._manage_checkpoints()

    def _manage_checkpoints(self, max_checkpoints=3):
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")))
        if len(checkpoint_files) > max_checkpoints:
            os.remove(checkpoint_files[0])

class TrainingConfig: 
    # Set config
    def __init__(self, **kwargs):
        default_lora_config = {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj"]
        }
        default_vocoder_config = {
            "codebook_size": 16384,
            "vocoder_layer": 1
        }
        default_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.lora_config = default_lora_config
        self.vocoder_config = default_vocoder_config
        
        # if 'lora_config' in kwargs, use it
        if 'lora_config' in kwargs:
            self.lora_config = kwargs['lora_config']
        if 'vocoder_config' in kwargs:
            self.vocoder_config = kwargs['vocoder_config']
        if 'model_name' in kwargs:
            self.model_name = kwargs['model_name']

def main():
    """
    Main function when using torchrun. Environment variables are set by torchrun.
    """
    from datasets import Dataset, load_dataset
    from itertools import islice
    from transformers import AutoTokenizer
    
    # For torchrun, we get rank and world_size from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize the distributed environment if world_size > 1
    if world_size > 1:
        setup_ddp()
        # Set device for this process
        torch.cuda.set_device(local_rank)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    
    training_config = TrainingConfig(
        model_name=args.model_name
    )
    lora_config = training_config.lora_config
    vocoder_config = training_config.vocoder_config
    model_name = training_config.model_name
    dataset_name = args.dataset_name
    
    if local_rank == 0:
        print("\n[DEBUG] Finished loading training config.")
        print("\n[DEBUG] Loading dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.from_disk:
        original_ds = load_from_disk(dataset_name)
        if local_rank == 0:
            print("[DEBUG] Successfully loaded from disk")
        ds = iter(original_ds.select_columns(["label_codec", "label_text"]))
        if local_rank == 0:
            print("[DEBUG] Change to iterable format")
    else:
        ds = load_dataset(dataset_name, streaming=True)["train"]
    
    def collect_and_save(iterable_dataset, num_samples=64):
        """
        Collects a specified number of samples from an IterableDataset and saves them as a Hugging Face Dataset.
        """
        collected_samples = list(islice(iterable_dataset, num_samples))
        hf_dataset = Dataset.from_list(collected_samples)
        return hf_dataset
    
    test_ds = collect_and_save(ds, num_samples=args.num_samples)
    if args.do_eval:
        # 0.9 as training set, 0.1 as validation set
        split_ds = test_ds.train_test_split(test_size=0.1)
        train_ds = split_ds['train']
        val_ds = split_ds['test']
    else:
        train_ds = test_ds
        val_ds = None

    if local_rank == 0:
        print("\n[DEBUG] Finished loading dataset.")
        print("\n[DEBUG] Initializing model...")
    
    # Each rank loads the model to CPU first
    base_model = AutoModelForCausalLM.from_pretrained( 
                model_name,
                device_map="cpu",
                torch_dtype="float",
                trust_remote_code=True
            )
    
    if local_rank == 0:
        print("\n[DEBUG] Finished initializing model.")
        print("\n[DEBUG] Initializing SpeechUnitModel...")
    
    speech_model = SpeechUnitModel(
        base_model,
        llama_layers=3,
        output_dim=vocoder_config["codebook_size"]+2, # Including BOS and EOS tokens
        num_heads=vocoder_config["vocoder_layer"],
        model_id=model_name,
    )
    
    if local_rank == 0:
        print("\n[DEBUG] Finished initializing SpeechUnitModel.")
    
    train_dataset = MimiUnitDataset(
        train_ds, 
        tokenizer, 
        num_layers=vocoder_config["vocoder_layer"], 
        codebook_size=vocoder_config["codebook_size"], 
        column_names=['label_text', 'label_codec']
    )
    
    if val_ds:
        val_dataset = MimiUnitDataset(
            val_ds, 
            tokenizer, 
            num_layers=vocoder_config["vocoder_layer"], 
            codebook_size=vocoder_config["codebook_size"], 
            column_names=['label_text', 'label_codec']
        )
    else:
        val_dataset = None
    
    if local_rank == 0:
        print("\n[DEBUG] Size of train_dataset: ", len(train_dataset))
        print("\n[DEBUG] Finished loading train_dataset.")
        print("\n[DEBUG] Initializing SpeechUnitTrainer...")
    
    # We pass local_rank and world_size to the trainer
    trainer = SpeechUnitTrainer(
        model=speech_model,
        lora_config=lora_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        use_wandb=args.use_wandb,
        project_name="speech_unit_training",
        checkpoint_dir=args.checkpoint_dir,
        codebook_size=vocoder_config["codebook_size"],
        vocoder_layer=vocoder_config["vocoder_layer"],
        rank=local_rank,
        world_size=world_size
    )
    
    if local_rank == 0:
        print("\n[DEBUG] Finished initializing SpeechUnitTrainer.")
        print("\n[DEBUG] All set! Training model...")
    
    trainer.train()
    
    if local_rank == 0:
        print("\n[DEBUG] Finished training model.")
    
    # Clean up the distributed environment
    if world_size > 1:
        dist.destroy_process_group()


    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="Allen172/GLM-4_codec_dataset")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_dir", type=str, default="ckpts/checkpoints")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--from_disk", action="store_true")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    args = parser.parse_args()


if __name__ == "__main__":
    main()