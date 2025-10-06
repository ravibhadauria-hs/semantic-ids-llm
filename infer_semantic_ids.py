#!/usr/bin/env python3
"""
Inference script to map item IDs to semantic IDs using trained RQ-VAE model.
This script loads a trained RQ-VAE checkpoint and generates semantic IDs for items.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.train_rqvae import RQVAE, RQVAEConfig, EmbeddingDataset
from src.device_manager import DeviceManager
from src.logger import setup_logger

logger = setup_logger("infer-semantic-ids", log_to_file=True)


class SemanticIDInferenceDataset(Dataset):
    """Dataset for inference that preserves item IDs alongside embeddings."""
    
    def __init__(self, embeddings_path: str, limit: Optional[int] = None):
        """Load embeddings and item IDs from parquet file.
        
        Args:
            embeddings_path: Path to parquet file with embeddings
            limit: Optional limit on number of items to load
        """
        logger.info(f"Loading embeddings from {embeddings_path}")
        df = pl.read_parquet(embeddings_path)
        
        if limit is not None:
            logger.info(f"Limiting to {limit} items")
            df = df.head(limit)
        
        # Extract embeddings and convert to tensor
        embeddings_list = df["embedding"].to_list()
        self.embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
        
        # Store item IDs for reference
        self.item_ids = df["parent_asin"].to_list()
        
        logger.info(f"Loaded {len(self.embeddings):,} embeddings of dimension {self.embeddings.shape[1]}")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.item_ids[idx]


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> Tuple[RQVAE, RQVAEConfig]:
    """Load RQ-VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint (allow pathlib.PosixPath for config compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        # Convert Path objects back to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        config = RQVAEConfig(**config_dict)
    else:
        # Fallback to default config if not found
        logger.warning("Config not found in checkpoint, using default config")
        config = RQVAEConfig()
    
    # Create model
    model = RQVAE(config)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Try loading the entire checkpoint as state dict
        state_dict = checkpoint
    
    # Handle compiled model state dict (remove _orig_mod. prefix)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        logger.info("Detected compiled model state dict, removing _orig_mod. prefix")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model config: {config.codebook_quantization_levels} levels, "
                f"{config.codebook_size} codes per level, "
                f"{config.codebook_embedding_dim}D vectors")
    
    return model, config


def generate_semantic_ids(
    model: RQVAE,
    data_loader: DataLoader,
    device: str,
    batch_size: int = 1024
) -> Tuple[List[str], np.ndarray]:
    """Generate semantic IDs for all items in the dataset.
    
    Args:
        model: Trained RQ-VAE model
        data_loader: DataLoader with embeddings and item IDs
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (item_ids, semantic_ids_array)
    """
    logger.info("Generating semantic IDs...")
    
    all_item_ids = []
    all_semantic_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (embeddings, item_ids) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            
            # Generate semantic IDs
            semantic_ids = model.encode_to_semantic_ids(embeddings)
            
            # Convert to numpy and store
            all_item_ids.extend(item_ids)
            all_semantic_ids.append(semantic_ids.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {len(all_item_ids):,} items")
    
    # Concatenate all semantic IDs
    semantic_ids_array = np.concatenate(all_semantic_ids, axis=0)
    
    logger.info(f"Generated semantic IDs for {len(all_item_ids):,} items")
    logger.info(f"Semantic ID shape: {semantic_ids_array.shape}")
    
    return all_item_ids, semantic_ids_array


def resolve_collisions(
    item_ids: List[str], 
    semantic_ids: np.ndarray, 
    config: RQVAEConfig
) -> Tuple[List[str], np.ndarray]:
    """Resolve semantic ID collisions by adding a 4th level code.
    
    Args:
        item_ids: List of item IDs
        semantic_ids: Array of semantic IDs (3 levels)
        config: Model configuration
        
    Returns:
        Tuple of (item_ids, semantic_ids_with_collision_resolution)
    """
    logger.info("Resolving semantic ID collisions...")
    
    # Create temporary DataFrame for collision analysis
    temp_data = {"parent_asin": item_ids}
    for level in range(config.codebook_quantization_levels):
        temp_data[f"semantic_id_{level}"] = semantic_ids[:, level]
    
    temp_df = pl.DataFrame(temp_data)
    
    # Group by the first 3 semantic IDs to find collisions
    collision_groups = (
        temp_df.with_row_index("original_order")
        .group_by([f"semantic_id_{i}" for i in range(config.codebook_quantization_levels)])
        .agg([pl.col("parent_asin"), pl.col("original_order"), pl.len().alias("group_size")])
        .sort([f"semantic_id_{i}" for i in range(config.codebook_quantization_levels)])
    )
    
    # Show collision statistics
    collision_counts = collision_groups.group_by("group_size").agg(pl.len().alias("n_groups")).sort("group_size")
    logger.info("Collision statistics:")
    for row in collision_counts.iter_rows():
        group_size, n_groups = row
        logger.info(f"  {n_groups:,} semantic ID combinations have {group_size} items")
    
    # Create 4th code for each item
    semantic_id_3_list = []
    
    for row in collision_groups.iter_rows():
        # Extract the first 3 semantic IDs and other data
        sem_ids = [row[i] for i in range(config.codebook_quantization_levels)]
        asins = row[config.codebook_quantization_levels]
        orders = row[config.codebook_quantization_levels + 1]
        group_size = row[config.codebook_quantization_levels + 2]
        
        # For each ASIN in this group, assign sequential 4th codes
        for idx, (asin, order) in enumerate(sorted(zip(asins, orders), key=lambda x: x[1])):
            semantic_id_3_list.append((order, idx))
    
    # Sort by original order and extract just the 4th codes
    semantic_id_3_list.sort(key=lambda x: x[0])
    semantic_id_3_values = [x[1] for x in semantic_id_3_list]
    
    # Create new semantic IDs array with 4th level
    semantic_ids_with_collision = np.column_stack([
        semantic_ids,
        np.array(semantic_id_3_values)
    ])
    
    # Verify uniqueness
    unique_ids = set()
    for ids in semantic_ids_with_collision:
        unique_ids.add(tuple(ids))
    
    unique_count = len(unique_ids)
    total_count = len(semantic_ids_with_collision)
    uniqueness_ratio = unique_count / total_count
    
    logger.info(f"After collision resolution:")
    logger.info(f"  Total unique semantic ID combinations: {unique_count:,} out of {total_count:,} items")
    logger.info(f"  Proportion of unique IDs: {uniqueness_ratio:.2%}")
    
    if uniqueness_ratio < 1.0:
        logger.warning(f"⚠️ Still have collisions after resolution: {uniqueness_ratio:.2%}")
    else:
        logger.info("✅ All semantic IDs are now unique!")
    
    return item_ids, semantic_ids_with_collision


def modify_output_path_with_config(output_path: str, config: RQVAEConfig, resolve_collisions_flag: bool) -> str:
    """Modify output path to include levels and codebook size information.
    
    Args:
        output_path: Original output path
        config: Model configuration
        resolve_collisions_flag: Whether collisions were resolved
        
    Returns:
        Modified output path with config info
    """
    from pathlib import Path
    
    # Parse the original path
    path = Path(output_path)
    
    # Create new filename with config info
    base_name = path.stem
    extension = path.suffix
    
    # Add config information to filename
    levels = config.codebook_quantization_levels
    codebook_size = config.codebook_size
    collision_suffix = "_collision_resolved" if resolve_collisions_flag else ""
    
    new_filename = f"{base_name}_L{levels}_C{codebook_size}{collision_suffix}{extension}"
    
    # Create new path
    new_path = path.parent / new_filename
    
    return str(new_path)


def save_results(
    item_ids: List[str],
    semantic_ids: np.ndarray,
    output_path: str,
    config: RQVAEConfig,
    resolve_collisions_flag: bool = True
) -> None:
    """Save semantic ID mapping to parquet file.
    
    Args:
        item_ids: List of item IDs
        semantic_ids: Array of semantic IDs
        output_path: Path to save results
        config: Model configuration
        resolve_collisions_flag: Whether to resolve collisions with 4th level code
    """
    # Resolve collisions if requested
    if resolve_collisions_flag and semantic_ids.shape[1] == config.codebook_quantization_levels:
        item_ids, semantic_ids = resolve_collisions(item_ids, semantic_ids, config)
    
    # Modify output path to include levels and codebook size
    output_path = modify_output_path_with_config(output_path, config, resolve_collisions_flag)
    logger.info(f"Saving results to {output_path}")
    
    # Create DataFrame with item IDs and semantic IDs
    data = {"parent_asin": item_ids}
    
    # Add each level of semantic IDs as separate columns
    for level in range(semantic_ids.shape[1]):
        data[f"semantic_id_level_{level}"] = semantic_ids[:, level]
    
    # Add combined semantic ID as string
    data["semantic_id"] = [
        "-".join(map(str, ids)) for ids in semantic_ids
    ]
    
    # Add metadata
    data["codebook_levels"] = [config.codebook_quantization_levels] * len(item_ids)
    data["codebook_size"] = [config.codebook_size] * len(item_ids)
    data["embedding_dim"] = [config.codebook_embedding_dim] * len(item_ids)
    data["collision_resolved"] = [resolve_collisions_flag] * len(item_ids)
    
    df = pl.DataFrame(data)
    df.write_parquet(output_path)
    
    logger.info(f"Saved {len(item_ids):,} semantic ID mappings to {output_path}")
    
    # Print sample results
    logger.info("Sample results:")
    logger.info(df.head(10))


def analyze_semantic_ids(semantic_ids: np.ndarray, config: RQVAEConfig) -> None:
    """Analyze the generated semantic IDs with comprehensive sanity checks.
    
    Args:
        semantic_ids: Array of semantic IDs
        config: Model configuration
    """
    logger.info("=== Semantic ID Analysis ===")
    
    # Basic statistics
    logger.info(f"Total items: {len(semantic_ids):,}")
    logger.info(f"Semantic ID shape: {semantic_ids.shape}")
    logger.info(f"Codebook levels: {config.codebook_quantization_levels}")
    logger.info(f"Codebook size per level: {config.codebook_size}")
    
    # Sanity check: Verify shape matches expected levels
    expected_levels = config.codebook_quantization_levels
    if semantic_ids.shape[1] != expected_levels:
        logger.error(f"❌ Shape mismatch: Expected {expected_levels} levels, got {semantic_ids.shape[1]}")
        return
    
    # Sanity check: Verify all codes are within valid range
    max_code = config.codebook_size - 1
    min_code = 0
    if np.any(semantic_ids < min_code) or np.any(semantic_ids > max_code):
        logger.error(f"❌ Invalid code range: Codes should be in [{min_code}, {max_code}]")
        invalid_mask = (semantic_ids < min_code) | (semantic_ids > max_code)
        invalid_count = np.sum(invalid_mask)
        logger.error(f"Found {invalid_count} invalid codes")
        return
    else:
        logger.info("✅ All codes within valid range")
    
    # Unique ID analysis
    unique_ids = set()
    for ids in semantic_ids:
        unique_ids.add(tuple(ids))
    
    unique_count = len(unique_ids)
    total_count = len(semantic_ids)
    uniqueness_ratio = unique_count / total_count
    
    logger.info(f"Unique semantic IDs: {unique_count:,}")
    logger.info(f"Uniqueness ratio: {uniqueness_ratio:.1%}")
    
    # Sanity check: Reasonable uniqueness ratio
    if uniqueness_ratio < 0.8:
        logger.warning(f"⚠️ Low uniqueness ratio: {uniqueness_ratio:.1%} (expected >80%)")
    else:
        logger.info("✅ Good uniqueness ratio")
    
    # Level-wise analysis
    for level in range(config.codebook_quantization_levels):
        level_ids = semantic_ids[:, level]
        unique_level_ids = len(set(level_ids))
        usage_ratio = unique_level_ids / config.codebook_size
        
        logger.info(f"Level {level}: {unique_level_ids:,} unique codes used "
                   f"out of {config.codebook_size} available ({usage_ratio:.1%})")
        
        # Sanity check: Reasonable codebook usage
        if usage_ratio < 0.5:
            logger.warning(f"⚠️ Low codebook usage at level {level}: {usage_ratio:.1%}")
        else:
            logger.info(f"✅ Good codebook usage at level {level}")
    
    # Collision analysis (following notebook pattern)
    from collections import Counter
    id_counts = Counter(tuple(ids) for ids in semantic_ids)
    collision_groups = Counter(id_counts.values())
    
    logger.info("Collision statistics:")
    for group_size in sorted(collision_groups.keys()):
        n_groups = collision_groups[group_size]
        logger.info(f"  {n_groups:,} semantic ID combinations have {group_size} items")
    
    # Most common semantic IDs
    most_common = id_counts.most_common(5)
    logger.info("Most common semantic IDs:")
    for ids, count in most_common:
        logger.info(f"  {ids}: {count} items")
    
    # Distribution analysis per level
    logger.info("Distribution statistics per level:")
    for level in range(config.codebook_quantization_levels):
        level_ids = semantic_ids[:, level]
        mean_val = np.mean(level_ids)
        std_val = np.std(level_ids)
        min_val = np.min(level_ids)
        max_val = np.max(level_ids)
        
        logger.info(f"  Level {level}: mean={mean_val:.1f}, std={std_val:.1f}, "
                   f"range=[{min_val}, {max_val}]")
    
    # Final sanity check summary
    logger.info("=== Sanity Check Summary ===")
    logger.info(f"✅ Shape correct: {semantic_ids.shape}")
    logger.info(f"✅ Code range valid: [{np.min(semantic_ids)}, {np.max(semantic_ids)}]")
    logger.info(f"✅ Uniqueness ratio: {uniqueness_ratio:.1%}")
    logger.info(f"✅ Codebook usage: {[len(set(semantic_ids[:, i])) for i in range(config.codebook_quantization_levels)]}")


def main():
    parser = argparse.ArgumentParser(description="Generate semantic IDs using trained RQ-VAE model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--embeddings", 
        type=str, 
        required=True,
        help="Path to embeddings parquet file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output path for semantic ID mapping (.parquet file)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1024,
        help="Batch size for inference (default: 1024)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of items to process (for testing)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--resolve-collisions", 
        action="store_true",
        help="Resolve semantic ID collisions by adding 4th level code"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device_manager = DeviceManager(logger)
        device = device_manager.device
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = SemanticIDInferenceDataset(args.embeddings, limit=args.limit)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device != "cpu",
        drop_last=False
    )
    
    # Generate semantic IDs
    item_ids, semantic_ids = generate_semantic_ids(
        model, data_loader, device, args.batch_size
    )
    
    # Analyze results
    analyze_semantic_ids(semantic_ids, config)
    
    # Save results
    save_results(item_ids, semantic_ids, args.output, config, args.resolve_collisions)
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
