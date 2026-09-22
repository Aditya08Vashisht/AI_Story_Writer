#!/bin/bash
#SBATCH --job-name=st-vllm
#SBATCH --output=logs/vllm-%j.out
#SBATCH --error=logs/vllm-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=htc
#SBATCH --qos=public
# htc caps at 4h; for a longer-lived server use:
#   sbatch --partition=public --qos=public --time=1-00:00:00 slurm/serve_vllm.sbatch

set -euo pipefail

module load mamba/latest
source activate storytutor

export PROJ=${PROJ:-/scratch/$USER/storytutor}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export TORCH_HOME=${TORCH_HOME:-/scratch/$USER/torch_cache}
export TOKENIZERS_PARALLELISM=false

# Apache-2.0. Drop to Qwen/Qwen3-8B-Instruct if this doesn't fit the node's VRAM.
MODEL=${VLLM_MODEL:-Qwen/Qwen3-14B-Instruct}
PORT=${VLLM_PORT:-8000}

cd "$PROJ"
mkdir -p logs

NODE=$(hostname)
echo "host        : $NODE"
echo "model       : $MODEL"
echo "port        : $PORT"
echo "started     : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Record where this server lives so a client on another node (e.g. the login
# node running the Flask app or an eval script) knows where to point.
ENDPOINT_FILE="$PROJ/slurm/vllm_endpoint.txt"
echo "http://${NODE}:${PORT}/v1" > "$ENDPOINT_FILE"
echo "endpoint    : $(cat "$ENDPOINT_FILE")"
echo "written to  : $ENDPOINT_FILE"
echo
echo "On another shell, point the app at this server with:"
echo "  export VLLM_BASE_URL=\$(cat $ENDPOINT_FILE)"
echo "  export VLLM_MODEL=$MODEL"

# --gpu-memory-utilization leaves headroom for the embedding model if this
# node is later reused for retrieval too; raise it if this node is
# dedicated to serving only.
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --dtype bfloat16
