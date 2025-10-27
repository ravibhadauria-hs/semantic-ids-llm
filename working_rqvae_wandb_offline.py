#!/usr/bin/env python3
"""
Custom RQ-VAE training script with wandb offline mode.
"""

from src.train_rqvae import RQVAEConfig, train_rqvae, RQVAE, EmbeddingDataset
from src.device_manager import DeviceManager
from src.logger import setup_logger
import torch
from torch.utils.data import DataLoader
import wandb
import os

# Set wandb to offline mode (no account needed)
os.environ["WANDB_MODE"] = "offline"

logger = setup_logger("train-rqvae-wandb-offline", log_to_file=True)

if __name__ == "__main__":
    # Customize your configuration here
    config = RQVAEConfig(
        category="jobs",
        codebook_quantization_levels=3,  # L3
        codebook_size=128,               # C128
        item_embedding_dim=768,
        batch_size=33668,
        num_epochs=1000,
        embeddings_path="data/output/jobs_with_bge_embeddings.parquet",
        
        # Improved learning rate schedule
        scheduler_type="cosine_with_warmup",
        warmup_start_lr=1e-8,
        max_lr=2e-4,        # Reduced from 3e-4
        min_lr=1e-5,        # Increased from 1e-6
        warmup_steps=200,  # Increased from 200
    )
    
    device_manager = DeviceManager(logger)
    device = device_manager.device

    # Initialize wandb in offline mode
    run_name = f"rqvae-L{config.codebook_quantization_levels}-C{config.codebook_size}-D{config.codebook_embedding_dim}"
    run = wandb.init(
        project="rqvae-jobs", 
        name=run_name, 
        config=config.__dict__,
        mode="offline"  # This ensures it works without login
    )
    
    logger.info("=== RQ-VAE Training Configuration ===")
    config.log_config()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = EmbeddingDataset(str(config.embeddings_path))
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train size: {len(train_dataset):,}, Val size: {len(val_dataset):,}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=device_manager.supports_pin_memory,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device_manager.supports_pin_memory,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Create model
    logger.info("Creating RQ-VAE model...")
    model = RQVAE(config)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    logger.info("Starting training...")
    train_rqvae(model=model, data_loader=train_loader, val_loader=val_loader, config=config, device=device)

    # Save final model
    final_path = config.checkpoint_dir / "final_model.pth"
    logger.info(f"Saving final model to {final_path}")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_path)

    logger.info("Training complete!")
    
    # Finish wandb run
    wandb.finish()