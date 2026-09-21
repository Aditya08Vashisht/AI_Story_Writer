#!/bin/bash
#SBATCH --job-name=st-index
#SBATCH --output=logs/build_index-%j.out
#SBATCH --error=logs/build_index-%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
# Sol moved publicly-available nodes to the "public" partition (2026).
# If this fails, list valid options with: sinfo -s -o "%20P %5a %10l %6D"
#SBATCH --partition=public
#SBATCH --qos=public

set -euo pipefail

module load mamba/latest
source activate storytutor

export PROJ=${PROJ:-/scratch/$USER/storytutor}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export TORCH_HOME=${TORCH_HOME:-/scratch/$USER/torch_cache}
export STORYTUTOR_DEVICE=cuda
export STORYTUTOR_EMBED_BATCH=${STORYTUTOR_EMBED_BATCH:-64}
export STORYTUTOR_MAX_SEQ_LEN=${STORYTUTOR_MAX_SEQ_LEN:-1024}
export TOKENIZERS_PARALLELISM=false

cd "$PROJ"
mkdir -p logs

echo "host    : $(hostname)"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/build_index.py --dataset-dir knowledge_base

echo "finished: $(date)"
