#!/bin/bash
#SBATCH --job-name=st-reval
#SBATCH --output=logs/retrieval_eval-%j.out
#SBATCH --error=logs/retrieval_eval-%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=htc
#SBATCH --qos=public

set -euo pipefail

module load mamba/latest
source activate storytutor

export PROJ=${PROJ:-/scratch/$USER/storytutor}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export STORYTUTOR_DEVICE=cuda
export STORYTUTOR_ENABLE_RERANK=${STORYTUTOR_ENABLE_RERANK:-1}
export TOKENIZERS_PARALLELISM=false

cd "$PROJ"
mkdir -p logs eval/reports

echo "=== WITH reranker ==="
STORYTUTOR_ENABLE_RERANK=1 python eval/run_retrieval_eval.py \
    --report eval/reports/retrieval_eval_rerank.json

echo
echo "=== WITHOUT reranker (fusion only) - is the reranker earning its cost? ==="
STORYTUTOR_ENABLE_RERANK=0 python eval/run_retrieval_eval.py \
    --report eval/reports/retrieval_eval_norerank.json
