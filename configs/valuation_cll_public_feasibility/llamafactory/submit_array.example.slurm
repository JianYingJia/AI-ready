#!/usr/bin/env bash
# Site-specific paths, partition and concurrency must be reviewed before submit.
# One array element owns one GPU and one immutable experiment trial.
#SBATCH --job-name=cll-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-251%8
#SBATCH --output=/shared/experiments/cll_lf_v1/slurm/%A_%a.out
#SBATCH --error=/shared/experiments/cll_lf_v1/slurm/%A_%a.err

set -euo pipefail

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate aiReadyLf

export AI_READY_ROOT=/shared/projects/ai-ready-benchmark
export BUNDLE_DIR=/shared/experiments/cll_lf_v1/training_bundle

# Slurm normally sets CUDA_VISIBLE_DEVICES.  The runner also verifies that its
# length equals the world_size frozen in server_profile.yaml.
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"

bash "${BUNDLE_DIR}/run_slurm_array.sh"
