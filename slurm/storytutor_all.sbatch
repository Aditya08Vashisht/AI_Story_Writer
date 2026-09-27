#!/bin/bash
# Everything, unattended: start the services, run every check, make the videos.
#   sbatch slurm/storytutor_all.sbatch
# Results: logs/storytutor_all-<job>.out, outputs/videos/, outputs/logs/.
#SBATCH --job-name=st-all
#SBATCH --output=logs/storytutor_all-%j.out
#SBATCH --error=logs/storytutor_all-%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=public
#SBATCH --qos=public

set -uo pipefail
module load mamba/latest
source activate storytutor

export PROJ=${PROJ:-/scratch/$USER/storytutor}
cd "$PROJ"
mkdir -p logs

bash scripts/sol.sh up     || exit 1
bash scripts/sol.sh test
bash scripts/sol.sh videos
bash scripts/sol.sh status
bash scripts/sol.sh down
