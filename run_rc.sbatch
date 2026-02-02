#!/bin/bash
#SBATCH -J rc_rf_gpu
#SBATCH -p gpuserver
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:2
#SBATCH --time=15:00:00
#SBATCH --output=logs/rc_rf_%j.out
#SBATCH --error=logs/rc_rf_%j.err

# ────────────────────────────────
# Setup ambiente
# ────────────────────────────────

# carica moduli se necessari (se già in conda puoi commentare)
# module load cuda/12.3

source /gpuserver/caputo/miniconda3/etc/profile.d/conda.sh
conda activate KC

# evita oversubscription CPU con multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# PyTorch multiprocessing (spawn già nel codice, ma meglio essere espliciti)
export PYTHONUNBUFFERED=1

# Debug CUDA se serve
# export CUDA_LAUNCH_BLOCKING=1

# ────────────────────────────────
# Run
# ────────────────────────────────

python multiGPUModels/randomCommitteeRF.py
