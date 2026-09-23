#!/bin/bash
#SBATCH --job-name=st-rageval
#SBATCH --output=logs/rag_eval-%j.out
#SBATCH --error=logs/rag_eval-%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=htc
#SBATCH --qos=public

set -euo pipefail
module load mamba/latest
source activate storytutor

export PROJ=${PROJ:-/scratch/$USER/storytutor}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export STORYTUTOR_DEVICE=cuda
export STORYTUTOR_ENABLE_RERANK=1
export TOKENIZERS_PARALLELISM=false

# Ollama must already be serving, and on THIS node -- a server bound to
# 127.0.0.1 on the login node is not reachable from here.
export PATH=/scratch/$USER/ollama/bin:$PATH
export OLLAMA_MODELS=/scratch/$USER/ollama/models
export OLLAMA_HOST=${OLLAMA_HOST:-http://127.0.0.1:11434}
export OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3:8b}

cd "$PROJ"
mkdir -p logs eval/reports

bash scripts/setup_ollama.sh serve || echo "ollama not started; will report passages_only"

python eval/run_rag_eval.py --limit ${EVAL_LIMIT:-60}
