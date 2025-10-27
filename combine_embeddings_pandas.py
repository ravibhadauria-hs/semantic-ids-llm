#!/usr/bin/env python3
"""
Combine multiple parquet files from GCS using pandas (handles Ray Data format).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def combine_parquet_files_pandas(input_dir: str, output_file: str):
    """Combine all parquet files in a directory into a single file using pandas."""
    
    # Find all parquet files
    parquet_files = glob.glob(f"{input_dir}/*.parquet")
    print(f"Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_dir}")
    
    # Read and combine all files
    print("Reading parquet files with pandas...")
    dfs = []
    for i, file in enumerate(parquet_files):
        print(f"Reading {i+1}/{len(parquet_files)}: {Path(file).name}")
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Combine all dataframes
    print("Combining dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Columns: {combined_df.columns.tolist()}")
    
    # Check the data structure
    print("\nData sample:")
    print(combined_df.head(3))
    
    # Check embedding dimension
    if 'embedding' in combined_df.columns:
        sample_embedding = combined_df['embedding'].iloc[0]
        if hasattr(sample_embedding, '__len__'):
            print(f"\nEmbedding dimension: {len(sample_embedding)}")
        else:
            print(f"\nEmbedding type: {type(sample_embedding)}")
    
    # Rename columns to match RQ-VAE expectations
    if 'entity_id' in combined_df.columns:
        combined_df = combined_df.rename(columns={'entity_id': 'parent_asin'})
        print("Renamed 'entity_id' to 'parent_asin'")
    
    # Convert embeddings to list format if they're numpy arrays
    if 'embedding' in combined_df.columns:
        print("Converting embeddings to list format...")
        combined_df['embedding'] = combined_df['embedding'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    # Save combined file
    print(f"\nSaving combined data to {output_file}...")
    combined_df.to_parquet(output_file, index=False)
    
    print(f"✅ Successfully saved {len(combined_df):,} embeddings to {output_file}")
    
    return combined_df

if __name__ == "__main__":
    input_dir = "data/output"
    output_file = "data/output/jobs_with_bge_embeddings.parquet"
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Combine files
    df = combine_parquet_files_pandas(input_dir, output_file)
    
    print(f"\n📊 Final dataset info:")
    print(f"   - Total items: {len(df):,}")
    print(f"   - Columns: {df.columns.tolist()}")
    print(f"   - File size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
    
    # Check embedding dimension
    if 'embedding' in df.columns:
        sample_embedding = df['embedding'].iloc[0]
        print(f"   - Embedding dimension: {len(sample_embedding)}")
