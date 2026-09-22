#!/bin/bash
# Create an isolated env for the vLLM server.
#
# vLLM pins torch tightly and installing it into the `storytutor` env would
# likely replace the torch build that the bge-m3 retrieval stack is already
# running on. The server is a separate process on a separate node, so it
# does not need to share an env with the app.
#
# Run on a LOGIN node (needs internet), not inside the job.
set -euo pipefail

module load mamba/latest
export PROJ=${PROJ:-/scratch/$USER/storytutor}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

mamba create -n storytutor-vllm python=3.11 -y
source activate storytutor-vllm
pip install -r "$PROJ/requirements-gpu.txt"

python -c "import vllm; print('vllm', vllm.__version__)"
echo "OK - now: sbatch slurm/serve_vllm.sbatch"
