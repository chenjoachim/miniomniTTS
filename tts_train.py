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

# import peft
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoConfig, get_cosine_schedule_with_warmup

import argparse

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
        val_per_epoch: int = 3,
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
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mimi_collate_fn, pin_memory=True
        )
        self.val_dataloader = None
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mimi_collate_fn, pin_memory=True
            )
        self.criterion = CrossEntropyLoss(ignore_index=0)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(len(self.train_dataloader) * self.num_epochs * warmup_ratio),
            num_training_steps=len(self.train_dataloader) * self.num_epochs,
        )
        
        self.val_per_epoch = val_per_epoch
        self.steps_per_epoch = len(self.train_dataloader)

        if use_wandb:
            wandb.init(project=project_name)

        if self.checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self):
        best_val_loss = float('inf')
        global_step = 0
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
                global_step += 1
                progress_bar.set_postfix({
                    'train_loss': train_loss / train_steps,
                    'grad_norm': grad_norm,
                    'clipped_grad_norm': clipped_grad_norm
                })

                if self.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'grad_norm': grad_norm,
                        'clipped_grad_norm': clipped_grad_norm,
                        'grad_norm_ratio': clipped_grad_norm / grad_norm if grad_norm > 0 else 0
                    })
                    
                if self.val_dataloader and step in [int(self.steps_per_epoch * i / self.val_per_epoch) for i in range(1, self.val_per_epoch)]:
                    val_loss = self.validate()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(f"best_model_epoch{epoch}_step{global_step}.pth")
                    print(f"Validation Loss: {val_loss:.4f}")
                    self.model.train()

            avg_train_loss = train_loss / train_steps

            if self.val_dataloader:
                val_loss = self.validate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model_epoch{epoch}_step{global_step}.pth")
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

            if self.checkpoint_dir and (epoch+1) % 20 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

    def _get_grad_norm(self) -> float:
        """Calculate the gradient norm for all parameters that require gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _compute_batch_loss(self, input_ids, labels, scheduled_sampling_prob=0.0):
        batch_size = input_ids.size(0)
        assert batch_size == labels.size(0), f"Input and labels must have the same batch size. Got input {batch_size} and label {labels.size(0)}"
        bos_labels = torch.full((batch_size, self.vocoder_layer, 1), self.codebook_size).to(self.device)
        eos_labels = torch.full((batch_size, self.vocoder_layer, 1), self.codebook_size+1).to(self.device)
        pad_labels = torch.full((batch_size, self.vocoder_layer, 1), 0).to(self.device)
        # generate_audio_ids = torch.cat([bos_labels, labels], dim=-1)
        if scheduled_sampling_prob > 0.0:
            # Initialize with padding/BOS
            current_input = bos_labels
            all_inputs = [current_input]
            
            # Step through sequence for scheduled sampling
            for t in range(labels.size(-1)):
                # Decide whether to use ground truth or prediction
                use_gt = torch.rand(batch_size, 1, 1).to(self.device) >= scheduled_sampling_prob
                use_gt = use_gt.expand(-1, self.vocoder_layer, -1)
                
                if t > 0:  # For tokens after the first one
                    # Get model prediction
                    with torch.no_grad():
                        temp_input = torch.cat(all_inputs, dim=-1)
                        temp_logits, _ = self.model(input_ids=input_ids, audio_ids=temp_input)
                        pred = torch.argmax(temp_logits[:, :, -1, :], dim=-1).unsqueeze(-1)
                    
                    # Mix ground truth and prediction
                    next_token = torch.where(use_gt, labels[:, :, t:t+1], pred)
                else:
                    # Always use ground truth for first token
                    next_token = labels[:, :, t:t+1]
                    
                all_inputs.append(next_token)
                
            generate_audio_ids = torch.cat(all_inputs, dim=-1)
        else:
            # Original behavior when scheduled sampling is disabled
            generate_audio_ids = torch.cat([bos_labels, labels], dim=-1)
        logits, text_logits = self.model(input_ids=input_ids, audio_ids=generate_audio_ids)
        # Drop first predict tokens
        logits = logits.view(-1, self.codebook_size+2)
        # print("logits shape: ", logits.shape)
        # print("labels shape: ", labels.shape)
        add_tensor = torch.zeros_like(labels)
        for i in range(1, self.vocoder_layer):
            add_tensor[:, i, :] = self.codebook_size * (i)
        labels = labels - add_tensor
        labels = torch.cat([labels, pad_labels], dim=-1) # eos_labels already in dataset
        audio_loss = self.criterion(logits, labels.view(-1))
        
        # Get text labels by shifting input_ids
        text_labels = input_ids[:, 1:]  # Shift right to create targets
        text_labels = text_labels.reshape(-1)
        text_logits = text_logits[:, :-1, :]  # Shift left to align with targets
        text_logits = text_logits.reshape(-1, text_logits.size(-1))
        text_loss = self.criterion(text_logits, text_labels)
        # Get the max out of text_logits
        # text_logits = torch.argmax(text_logits, dim=-1)
        # print("LABEL:",self.model.tokenizer.decode(text_labels[:10].cpu().numpy()))
        # print("PRED:",self.model.tokenizer.decode(text_logits[:10].cpu().numpy()))

        # print("AUDIO LABEL:", labels[:10].cpu().numpy())
        # print("AUDIO PRED:", torch.argmax(logits, dim=-1)[:10].cpu().numpy())
        # Combine audio and text loss
        total_loss = audio_loss + text_loss
        return total_loss
        
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
        # Only save trainable component
        model_state_dict = self.model.state_dict()
        # Get names of trainable parameters
        trainable_param_names = [name for name, param in self.model.named_parameters() if param.requires_grad]

        # Filter state dict using those names
        trainable_state_dict = {k: v for k, v in model_state_dict.items() if (k in trainable_param_names or "audio_embed" in k)}
        print("Trainable state dict keys:")
        print(trainable_state_dict.keys())
        torch.save(trainable_state_dict, checkpoint_path)
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
    from datasets import Dataset, load_dataset
    from itertools import islice
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="Allen172/GLM-4_codec_dataset")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_dir", type=str, default="ckpts/checkpoints")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--from_disk", action="store_true")
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--use_full_model", action="store_true")
    parser.add_argument("--fully_FT", action="store_true")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_per_epoch", type=int, default=3)
    parser.add_argument("--codebook_ckpt", type=str, default=None)
    args = parser.parse_args()
    
    
    training_config = TrainingConfig(
        model_name=args.model_name
    )
    lora_config = training_config.lora_config
    vocoder_config = training_config.vocoder_config
    model_name = training_config.model_name
    dataset_name = args.dataset_name
    
    print("\n[DEBUG] Finished loading training config.")
    print("\n[DEBUG] Loading dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.from_disk:
        original_ds = load_from_disk(dataset_name)
        print("[DEBUG] Successfully loaded from disk")
        ds = iter(original_ds.select_columns(["label_codec", "label_text"]))
        print("[DEBUG] Change to iterable format")
    else:
        original_ds = load_dataset(dataset_name, streaming=True)["train"]
        ds = iter(original_ds.select_columns(["label_codec", "label_text"]))
        print("[DEBUG] Change to iterable format with selected columns")
    
    #def collect_and_save(iterable_dataset, num_samples=64):
        """
        Collects a specified number of samples from an IterableDataset and saves them as a Hugging Face Dataset.
        """
    #    collected_samples = list(islice(iterable_dataset, num_samples))
    #    hf_dataset = Dataset.from_list(collected_samples)
    def collect_and_save(iterable_dataset, num_samples=64):
        """
        Collects a specified number of samples from an IterableDataset and saves them as a Hugging Face Dataset.
        Includes a progress bar to track completion.
        """
        # Create an empty list to store samples
        collected_samples = []
    
        # Use tqdm to create a progress bar
        with tqdm(total=num_samples, desc="Collecting samples") as pbar:
            for i, sample in enumerate(iterable_dataset):
                if i >= num_samples:
                    break
                collected_samples.append(sample)
                pbar.update(1)
    
        # Convert to Hugging Face dataset
        hf_dataset = Dataset.from_list(collected_samples)
        return hf_dataset
    
    test_ds = collect_and_save(ds, num_samples=args.num_samples)
    if args.do_eval:
        # 0.9 as training set, 0.1 as validation set
        split_ds = test_ds.train_test_split(test_size=args.test_ratio, shuffle=True)
        train_ds = split_ds['train']
        val_ds = split_ds['test']
    else:
        train_ds = test_ds
        val_ds = None

    # def collect_and_save(iterable_dataset, num_samples=256):
    #     collected_samples = list(islice(iterable_dataset, num_samples))
    #     return Dataset.from_list(collected_samples)
    # ds1 = load_from_disk('../Soundon-TTS-preprocessing/_machine_nonoverlap_0_100_mimi_final')
    # ds2 = load_from_disk('../Soundon-TTS-preprocessing/dialogue_chinese_llama31_70B_user_long_mimi_final')
    # ds3 = load_from_disk('../Soundon-TTS-preprocessing/TW_Attraction_mimi_dataset')
    # test_ds = concatenate_datasets([ds1, ds2, ds3])
    # print(test_ds)
    # test_ds = load_from_disk('./mimiTTS_dataset')


    # test_ds = collect_and_save(ds['train'], num_samples=10000)
    print("\n[DEBUG] Finished loading dataset.")
    print("\n[DEBUG] Initializing model...")
    base_model = AutoModelForCausalLM.from_pretrained( 
                model_name,
                device_map="cpu",
                torch_dtype="float",
                trust_remote_code=True,  
                # attn_implementation="flash_attention_2"
            )
    print("\n[DEBUG] Finished initializing model.")
    print("\n[DEBUG] Initializing SpeechUnitModel...")
    speech_model = SpeechUnitModel(
        base_model,
        llama_layers=args.num_layers,
        output_dim=vocoder_config["codebook_size"]+2, # Including BOS and EOS tokens
        num_heads=vocoder_config["vocoder_layer"],
        model_id=model_name,
        use_full_model=args.use_full_model,
        preserve_lm_head=args.use_full_model,
        fully_FT=args.fully_FT,
        codebook_ckpt=args.codebook_ckpt,
    ).to('cuda')
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
    print("\n[DEBUG] Size of train_dataset: ", len(train_dataset))
    print("\n[DEBUG] Finished loading train_dataset.")
    print("\n[DEBUG] Initializing SpeechUnitTrainer...")
    trainer = SpeechUnitTrainer(
        model=speech_model,
        lora_config=lora_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=True,
        project_name="speech_unit_training",
        checkpoint_dir=args.checkpoint_dir,
        codebook_size=16384,
        vocoder_layer=1,
        val_per_epoch=args.val_per_epoch,
    )
    print("\n[DEBUG] Finished initializing SpeechUnitTrainer.")
    print("\n[DEBUG] All set! Training model...")
    trainer.train()
    print("\n[DEBUG] Finished training model.")


if __name__ == "__main__":
    main()

