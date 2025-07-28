import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from accelerate import Accelerator
from config import DFLMConfig
from model import DFLM
import tempfile
import hashlib
import pickle
import wandb
import numpy as np
from typing import List, Dict, Any
from loss import MixturePathGeneralizedKL, MixtureDiscreteProbPath, SimplifiedFlowLoss


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
                        
                    # print(text)
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
                batch_size=64,  # Increased batch size for better efficiency
                remove_columns=self.dataset.column_names  # Remove original columns to save memory
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
        # Return dictionary format as expected by the training loop
        return self.processed_data[idx]


def collate_fn(batch):
    """Custom collate function for dynamic batching (optional optimization)"""
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def evaluate_model(model, val_loader, loss_fn, num_timesteps, time_scheduler, accelerator, max_eval_batches=50):
    """Evaluate model on validation set"""
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
            
            # Use the flow loss for evaluation
            if isinstance(loss_fn, MixturePathGeneralizedKL):
                loss = loss_fn(logits, target, noised_input_ids, time_t)
            else:
                loss = loss_fn(logits, target, noised_input_ids, time_t)
            
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    return total_loss / max(total_batches, 1)


def calculate_perplexity(model, text_samples: List[str]) -> float:
    """Calculate perplexity on generated text samples"""
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in text_samples:
            # Tokenize the text
            tokens = model.tokenizer(text, return_tensors="pt", padding=False, truncation=False)
            input_ids = tokens["input_ids"].to(model.device)
            
            if input_ids.shape[1] <= 1:  # Skip empty or single token sequences
                continue
                
            # Calculate log likelihood for each position
            for i in range(1, input_ids.shape[1]):
                context = input_ids[:, :i]
                target = input_ids[:, i]
                
                # For simplicity, we'll use a basic forward pass
                # Note: This is a simplified perplexity calculation
                # In practice, you might want to use a more sophisticated approach
                attention_mask = torch.ones_like(context, dtype=torch.bool)
                
                # Use model to predict next token
                timesteps = torch.zeros(context.shape[0], device=model.device)  # Use t=0 for clean generation
                logits = model(context, attention_mask, timesteps)
                
                # Get probability of target token
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                total_log_likelihood += log_probs[0, target].item()
                total_tokens += 1
    
    model.train()
    
    if total_tokens == 0:
        return float('inf')
    
    avg_log_likelihood = total_log_likelihood / total_tokens
    perplexity = torch.exp(-torch.tensor(avg_log_likelihood)).item()
    
    return perplexity


def generate_and_evaluate(model, accelerator, num_samples=5, sequence_length=256, prompts=None):
    """Generate samples and calculate metrics"""
    if prompts is None:
        prompts = [
            "Human: Write a poem about nature.\nAssistant: ",
            "Human: Tell me a story about adventure.\nAssistant: ",
            "Human: Describe a beautiful sunset.\nAssistant: ",
        ]
    
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts[:num_samples]:
            try:
                # Generate sample using the model's sampling method
                samples = model.sample(
                    num_sampling_steps=64,  # Reduced for faster evaluation
                    num_samples=1,
                    sequence_length=sequence_length,
                    yield_intermediate=False,
                    temperature=0.7,
                    stochasticity=3.0,
                    prompt=prompt
                )
                
                # Get the final sample
                final_sample = None
                for sample in samples:
                    final_sample = sample
                
                if final_sample is not None:
                    token_ids = final_sample[1]
                    text = model.tokenizer.decode(token_ids[0], skip_special_tokens=True)
                    generated_texts.append(text)
                    
                    if accelerator.is_main_process:
                        print(f"Generated sample:\n{text[:200]}...\n")
                        
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Error generating sample: {e}")
                generated_texts.append("")
    
    model.train()
    
    # Calculate perplexity on generated samples
    valid_texts = [text for text in generated_texts if len(text.strip()) > 0]
    if valid_texts:
        try:
            ppl = calculate_perplexity(model, valid_texts)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Error calculating perplexity: {e}")
            ppl = float('inf')
    else:
        ppl = float('inf')
    
    return {
        "generated_texts": generated_texts,
        "perplexity": ppl,
        "num_valid_samples": len(valid_texts)
    }


def main(
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
    use_flow_loss=True,
):
    # Initialize wandb
    if run_name is None:
        run_name = f"dflm_bs{batch_size}_lr{learning_rate}_wd{weight_decay}"
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",  # Enable mixed precision training
        log_with=["wandb"],  # Use wandb instead of tensorboard
        project_dir="./logs"
    )
    
    # Initialize wandb only on main process
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
                "use_flow_loss": use_flow_loss,
                "model_params": model.analyze_params(),
            },
            init_kwargs={"wandb": {"name": run_name}}
        )
    
    # Optimized DataLoader configuration
    is_cuda = accelerator.device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=True,
        pin_memory=is_cuda,  # Pin memory for GPU training
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
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
    
    # Initialize the loss function
    if use_flow_loss:
        # Create probability path for discrete flow matching
        path = MixtureDiscreteProbPath()
        loss_fn = MixturePathGeneralizedKL(path=path, reduction="mean")
        if accelerator.is_main_process:
            print("Using MixturePathGeneralizedKL loss")
    else:
        # Use simplified flow loss as fallback
        loss_fn = SimplifiedFlowLoss(reduction="mean")
        if accelerator.is_main_process:
            print("Using SimplifiedFlowLoss")
    
    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader, loss_fn = accelerator.prepare(
        model, optimizer, train_loader, val_loader, loss_fn
    )
    
    total_steps = len(train_loader) * epochs
    
    # Only show progress bar on main process
    if accelerator.is_main_process:
        pbar = tqdm(total=max_steps or total_steps)
    
    # Move to device for efficiency
    time_scheduler = torch.linspace(
        1 / num_timesteps,
        1,
        steps=num_timesteps,
        dtype=torch.float32,
        device=accelerator.device,
    )
    
    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            # Move batch to device efficiently (accelerator handles this automatically)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].bool()
            
            timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
            time_t = time_scheduler[timesteps]
            
            noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
            
            logits = model(noised_input_ids, attention_mask, time_t)
            
            target = input_ids.clone()
            
            # Only calculate loss on tokens that were masked
            target[noised_input_ids != model.mask_token_id] = -100
            target[attention_mask == 0] = -100  
            
            if accelerator.is_main_process and step % 100 == 0:  # Reduce print frequency
                print(f"logits: {logits.shape}")
                print(f"target: {target.shape}")
                print(f"noised_input_ids: {noised_input_ids.shape}")
                print(f"time_t: {time_t.shape}")
            
            # Use the flow loss
            if isinstance(loss_fn, MixturePathGeneralizedKL):
                loss = loss_fn(logits, target, noised_input_ids, time_t)
            else:
                loss = loss_fn(logits, target, noised_input_ids, time_t)
            
            # Use accelerator's backward pass
            accelerator.backward(loss)
            
            # Calculate gradient norm before clipping
            grad_norm = 0.0
            if accelerator.sync_gradients:
                # Apply gradient clipping
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Calculate gradient norm across all parameters
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
            
            # Log metrics
            metrics = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/step": step,
                "train/epoch": epoch,
            }
            
            # Update progress bar only on main process
            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                pbar.update(1)
                
                # Log to wandb
                accelerator.log(metrics, step=step)
            
            step += 1
            if max_steps and step >= max_steps:
                break
                
            # Evaluation loop
            if step % eval_every == 0:
                if accelerator.is_main_process:
                    print(f"\nRunning evaluation at step {step}...")
                
                val_loss = evaluate_model(model, val_loader, loss_fn, num_timesteps, time_scheduler, accelerator)
                
                eval_metrics = {
                    "eval/loss": val_loss,
                    "eval/step": step,
                }
                
                if accelerator.is_main_process:
                    print(f"Validation loss: {val_loss:.4f}")
                    accelerator.log(eval_metrics, step=step)
            
            # Generation evaluation loop
            if step % gen_eval_every == 0 and step > 0:
                if accelerator.is_main_process:
                    print(f"\nRunning generation evaluation at step {step}...")
                
                gen_results = generate_and_evaluate(model, accelerator)
                
                gen_metrics = {
                    "gen_eval/perplexity": gen_results["perplexity"],
                    "gen_eval/num_valid_samples": gen_results["num_valid_samples"],
                    "gen_eval/step": step,
                }
                
                if accelerator.is_main_process:
                    print(f"Generation perplexity: {gen_results['perplexity']:.4f}")
                    accelerator.log(gen_metrics, step=step)
        
        # Save model only on main process
        if accelerator.is_main_process:
            # Unwrap model for saving
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), f"model_epoch_{epoch}.pth")
        
        if max_steps and step >= max_steps:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), f"model_final_step_{step}.pth")
            break
    
    if accelerator.is_main_process:
        pbar.close()
        accelerator.end_training()


if __name__ == "__main__":
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
    
    # model.load_state_dict(torch.load("/home/sang/work/discrete-flow-llm/model_epoch_9.pth"))
    
    print(model.analyze_params())
    
    tokenizer = AutoTokenizer.from_pretrained("../smol_tokenizer/")    
    train_dataset = SmolTextDataset(
        dataset_name="tatsu-lab/alpaca",
        text_cols=["text"],
        tokenizer=tokenizer,
        max_length=512,
        split="train",
        use_cache=True,
        add_eos=True  # Add EOS token to training text
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
    
    main(
        model,
        train_dataset,
        val_dataset,
        batch_size=32,
        num_workers=4,
        eval_every=500,  # Evaluate every 500 steps
        gen_eval_every=2000,  # Generate and evaluate every 2000 steps
        num_timesteps=512,
        epochs=100,
        learning_rate=1e-4,
        weight_decay=1e-2,
        max_steps=20000,
        max_grad_norm=1.0,
        project_name="discrete-flow-llm",
        run_name="dflm_with_flow_loss_v1",
        use_flow_loss=True  # Set to False to use simplified loss
    )