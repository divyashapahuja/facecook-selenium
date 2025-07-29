"""
Final training code using Flow Matching loss with your existing time_scheduler.
This shows the complete integration with minimal changes to your original code.
"""

import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from accelerate import Accelerator
# from config import DFLMConfig  # Your existing imports
# from model import DFLM
import tempfile
import hashlib
import pickle
import wandb
import numpy as np
from typing import List, Dict, Any

# Import the new loss functions
from loss_final import MixturePathGeneralizedKL, MixtureDiscreteProbPath, DiscreteDiffusionLoss

# Your existing SmolTextDataset class (unchanged)
class SmolTextDataset(Dataset):
    def __init__(self, dataset_name, text_cols, tokenizer, max_length=1024, split="train", use_cache=True, add_eos=True, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cache = use_cache
        self.add_eos = add_eos
        self.max_samples = max_samples
        
        # Create cache key for this specific configuration
        cache_key = hashlib.md5(
            f"{dataset_name}_{'-'.join(text_cols)}_{max_length}_{split}_{tokenizer.name_or_path}_{add_eos}".encode()
        ).hexdigest()[:12]
        
        self.cache_path = os.path.join(tempfile.gettempdir(), f"smol_dataset_cache_{cache_key}.pkl")
        
        # Try to load from cache first
        if use_cache and os.path.exists(self.cache_path):
            print(f"Loading dataset from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print("Processing dataset...")
            self.dataset = load_dataset(dataset_name, split=split)
            
            def tokenize_function(examples):
                texts = []
                
                for i in range(len(examples[text_cols[0]])):
                    text = "".join(examples[col][i] for col in text_cols)
                    # Add EOS token if specified
                    if self.add_eos:
                        text = text + self.tokenizer.eos_token
                        
                    texts.append(text)
                
                # Use return_tensors='pt' for better performance
                tokenized = self.tokenizer(
                    texts, 
                    max_length=self.max_length, 
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                
                return {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"]
                }
            
            if self.max_samples is not None:
                self.dataset = self.dataset.select(range(self.max_samples))
                
            # Process the dataset
            processed_dataset = self.dataset.map(
                tokenize_function, 
                batched=True, 
                num_proc=os.cpu_count(),
                batch_size=64,
                remove_columns=self.dataset.column_names
            )
            
            # Convert to list for faster indexing
            self.processed_data = []
            for item in processed_dataset:
                self.processed_data.append({
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"]
                })
            
            # Cache for future use
            if use_cache:
                print(f"Caching processed dataset to: {self.cache_path}")
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.processed_data, f)

    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch):
    """Custom collate function for dynamic batching (optional optimization)"""
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def evaluate_model_flow_matching(model, val_loader, num_timesteps, time_scheduler, path, loss_fn, accelerator, max_eval_batches=50):
    """Evaluate model on validation set using Flow Matching loss"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_eval_batches:
                break
                
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].bool()
            
            # Sample timesteps
            timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
            time_t = time_scheduler[timesteps]
            
            # Flow Matching approach
            x_0 = torch.full_like(input_ids, model.mask_token_id)
            path_sample = path.sample(x_0=x_0, x_1=input_ids, timesteps=timesteps)
            noised_input_ids = path_sample.x_t
            
            logits = model(noised_input_ids, attention_mask, time_t)
            
            loss = loss_fn(
                logits=logits,
                x_1=input_ids,
                x_t=noised_input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
            
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    return total_loss / max(total_batches, 1)


def evaluate_model_discrete(model, val_loader, num_timesteps, time_scheduler, loss_fn, accelerator, max_eval_batches=50):
    """Evaluate model on validation set using discrete diffusion loss"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_eval_batches:
                break
                
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].bool()
            
            timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
            time_t = time_scheduler[timesteps]
            
            noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
            
            logits = model(noised_input_ids, attention_mask, time_t)
            
            target = input_ids.clone()
            target[noised_input_ids != model.mask_token_id] = -100
            target[attention_mask == 0] = -100
            
            loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
            
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    return total_loss / max(total_batches, 1)


def main_with_flow_matching(
    model,
    train_dataset,
    val_dataset,
    batch_size=128,
    num_workers=4,
    eval_every=1000,
    gen_eval_every=5000,
    num_timesteps=128,
    epochs=10,
    learning_rate=1e-3,
    weight_decay=1e-2,
    max_steps=None,
    max_grad_norm=1.0,
    project_name="discrete-flow-llm",
    run_name=None,
    loss_type="flow_matching",  # "flow_matching" or "discrete"
):
    """
    Main training function using Flow Matching loss with your existing time_scheduler.
    
    Args:
        loss_type: "flow_matching" for Flow Matching loss, "discrete" for discrete diffusion loss
    """
    # Initialize wandb
    if run_name is None:
        run_name = f"dflm_bs{batch_size}_lr{learning_rate}_wd{weight_decay}_{loss_type}"
    
    # Initialize accelerator (unchanged)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with=["wandb"],
        project_dir="./logs"
    )
    
    # Initialize wandb only on main process (unchanged)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=project_name,
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_timesteps": num_timesteps,
                "epochs": epochs,
                "max_steps": max_steps,
                "max_grad_norm": max_grad_norm,
                "loss_type": loss_type,
                # "model_params": model.analyze_params(),
            },
            init_kwargs={"wandb": {"name": run_name}}
        )
    
    # Optimized DataLoader configuration (unchanged)
    is_cuda = accelerator.device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=True,
        pin_memory=is_cuda,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False,
        pin_memory=is_cuda,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Prepare everything with accelerator (unchanged)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    total_steps = len(train_loader) * epochs
    
    # Only show progress bar on main process (unchanged)
    if accelerator.is_main_process:
        pbar = tqdm(total=max_steps or total_steps)
    
    # YOUR EXISTING TIME SCHEDULER (no changes!)
    time_scheduler = torch.linspace(
        1 / num_timesteps,
        1,
        steps=num_timesteps,
        dtype=torch.float32,
        device=accelerator.device,
    )
    
    # CREATE LOSS COMPONENTS ONCE (before training loop)
    if loss_type == "flow_matching":
        path = MixtureDiscreteProbPath(time_scheduler)
        loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
        if accelerator.is_main_process:
            print("Using Flow Matching loss")
    else:  # discrete
        loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")
        path = None
        if accelerator.is_main_process:
            print("Using Discrete Diffusion loss")
    
    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].bool()
            
            # Your existing timestep sampling (no changes!)
            timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
            time_t = time_scheduler[timesteps]  # For your model
            
            if loss_type == "flow_matching":
                # Flow Matching approach
                x_0 = torch.full_like(input_ids, model.mask_token_id)
                path_sample = path.sample(x_0=x_0, x_1=input_ids, timesteps=timesteps)
                noised_input_ids = path_sample.x_t
                
                logits = model(noised_input_ids, attention_mask, time_t)
                
                loss = loss_fn(
                    logits=logits,
                    x_1=input_ids,
                    x_t=noised_input_ids,
                    timesteps=timesteps,
                    attention_mask=attention_mask
                )
            else:
                # Discrete diffusion approach (your original method)
                noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
                
                logits = model(noised_input_ids, attention_mask, time_t)
                
                target = input_ids.clone()
                target[noised_input_ids != model.mask_token_id] = -100
                target[attention_mask == 0] = -100
                
                loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
            
            # Rest of your training loop is UNCHANGED
            accelerator.backward(loss)
            
            # Calculate gradient norm before clipping (unchanged)
            grad_norm = 0.0
            if accelerator.sync_gradients:
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                total_norm = 0.0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                grad_norm = total_norm ** (1. / 2) if param_count > 0 else 0.0
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Log metrics (unchanged)
            metrics = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/step": step,
                "train/epoch": epoch,
                "train/loss_type": loss_type,
            }
            
            # Update progress bar only on main process (unchanged)
            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "loss_type": loss_type
                })
                pbar.update(1)
                accelerator.log(metrics, step=step)
            
            step += 1
            if max_steps and step >= max_steps:
                break
                
            # Evaluation loop
            if step % eval_every == 0:
                if accelerator.is_main_process:
                    print(f"\nRunning evaluation at step {step}...")
                
                if loss_type == "flow_matching":
                    val_loss = evaluate_model_flow_matching(
                        model, val_loader, num_timesteps, time_scheduler, 
                        path, loss_fn, accelerator, max_eval_batches=50
                    )
                else:
                    val_loss = evaluate_model_discrete(
                        model, val_loader, num_timesteps, time_scheduler,
                        loss_fn, accelerator, max_eval_batches=50
                    )
                
                eval_metrics = {
                    "eval/loss": val_loss,
                    "eval/step": step,
                }
                
                if accelerator.is_main_process:
                    print(f"Validation loss: {val_loss:.4f}")
                    accelerator.log(eval_metrics, step=step)
            
            # Generation evaluation loop (unchanged)
            if step % gen_eval_every == 0 and step > 0:
                if accelerator.is_main_process:
                    print(f"\nRunning generation evaluation at step {step}...")
                
                # Your existing generation evaluation code here
                # gen_results = generate_and_evaluate(model, accelerator)
                # ... rest unchanged
        
        # Save model only on main process (unchanged)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), f"model_epoch_{epoch}_{loss_type}.pth")
        
        if max_steps and step >= max_steps:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), f"model_final_step_{step}_{loss_type}.pth")
            break
    
    if accelerator.is_main_process:
        pbar.close()
        accelerator.end_training()


# Example usage
if __name__ == "__main__":
    # Uncomment and modify with your actual model setup:
    """
    config = DFLMConfig(
        dim=1024,
        depth=12,
        heads=8,
        dim_head=128,
        dropout=0.1,
        text_dim=512,
        tokenizer="../smol_tokenizer/",
        max_length=512
    )

    model = DFLM(config)
    
    tokenizer = AutoTokenizer.from_pretrained("../smol_tokenizer/")    
    train_dataset = SmolTextDataset(
        dataset_name="tatsu-lab/alpaca",
        text_cols=["text"],
        tokenizer=tokenizer,
        max_length=512,
        split="train",
        use_cache=True,
        add_eos=True
    )
    
    val_dataset = SmolTextDataset(
        dataset_name="tatsu-lab/alpaca",
        text_cols=["text"],
        tokenizer=tokenizer,
        max_length=512,
        split="train",
        use_cache=False,
        add_eos=True,
        max_samples=50
    )
    
    # Option 1: Flow Matching loss
    main_with_flow_matching(
        model,
        train_dataset,
        val_dataset,
        batch_size=32,
        num_timesteps=512,
        loss_type="flow_matching",  # Use Flow Matching
        learning_rate=1e-4,
        weight_decay=1e-2,
        max_steps=20000,
        project_name="discrete-flow-llm",
        run_name="flow_matching_test"
    )
    
    # Option 2: Discrete diffusion loss (closer to your original)
    main_with_flow_matching(
        model,
        train_dataset,
        val_dataset,
        batch_size=32,
        num_timesteps=512,
        loss_type="discrete",  # Use discrete loss
        learning_rate=1e-4,
        weight_decay=1e-2,
        max_steps=20000,
        project_name="discrete-flow-llm",
        run_name="discrete_test"
    )
    """
    
    print("Training script ready!")
    print("Uncomment the example usage section and modify with your model setup.")
    print("\nKey changes from your original code:")
    print("1. Import: from loss_final import MixturePathGeneralizedKL, MixtureDiscreteProbPath, DiscreteDiffusionLoss")
    print("2. Create loss components ONCE before training loop")
    print("3. Choose loss_type='flow_matching' or 'discrete'")
    print("4. Everything else stays the same!")