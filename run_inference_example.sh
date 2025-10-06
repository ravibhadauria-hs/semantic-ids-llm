#!/bin/bash
# Example script to run semantic ID inference

# Set paths
CHECKPOINT_PATH="checkpoints/rqvae/best_model.pth"
EMBEDDINGS_PATH="data/output/jobs_with_bge_embeddings.parquet"
OUTPUT_PATH="data/output/jobs_semantic_ids.parquet"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la checkpoints/rqvae/*.pth
    exit 1
fi

# Check if embeddings exist
if [ ! -f "$EMBEDDINGS_PATH" ]; then
    echo "Error: Embeddings not found at $EMBEDDINGS_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Running semantic ID inference..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Embeddings: $EMBEDDINGS_PATH"
echo "Output: $OUTPUT_PATH"

# Run inference with collision resolution
uv run python infer_semantic_ids.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --embeddings "$EMBEDDINGS_PATH" \
    --output "$OUTPUT_PATH" \
    --batch-size 1024 \
    --device cpu \
    --resolve-collisions

echo "Inference complete! Results saved to $OUTPUT_PATH"
