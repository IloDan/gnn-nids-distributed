#!/bin/bash
#SBATCH -J rf_train
#SBATCH -p gpuserver
#SBATCH -N 1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /gpuserver/caputo/miniconda3/etc/profile.d/conda.sh
conda activate KC_rapids_2312_cuda12

cd /gpuserver/caputo/keep_calm
python -u multiGPUModels/randomforest_cuda.py
