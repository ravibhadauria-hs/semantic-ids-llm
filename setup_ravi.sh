#!/bin/bash

# Setup script for Ravi
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# add to path
export PATH="/home/ray/.local/bin:$PATH"
uv --version
# install and configure uv
uv sync
# verify environment
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); import transformers; print(f'Transformers version: {transformers.__version__}'); import wandb; print(f'Wandb version: {wandb.__version__}')"
# update this line because pkg_resources is set to be deprecated soon
uv run python -c "import sys; print('Python path:'); [print(p) for p in sys.path]; print('\nInstalled packages:'); import pkg_resources; [print(d.project_name) for d in pkg_resources.working_set if 'semantic' in d.project_name.lower()]"
# verify if installed
uv run python -c "import device_manager; print('✅ device_manager module imported successfully!')"

# download data
gsutil -m cp -r "gs://handshake-production-data-scratch/recommendations/jobs/embeddings/bge_base/pretrained_embeddings_scheduled__2025-10-03T230000+0000/entity_type=job_id/*" data/output/

# modify data to create RQ-VAE training file
uv run python combine_embeddings_pandas.py