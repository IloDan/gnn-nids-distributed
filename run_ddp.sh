#!/bin/bash

SEEDS=(42 123 999 2025 7)
PROTOCOL=("all" "ToN")
NUM_GPUS=2
PORT=12355

source ~/miniconda3/etc/profile.d/conda.sh  # oppure path corretto alla tua installazione
conda activate KC

for SEED in "${SEEDS[@]}"
do
  echo "🚀 Avvio esecuzione con seed $SEED e protocollo $PROTOCOL"
  SEED=$SEED PROTOCOLS=$PROTOCOL torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT GNN/train_ddp.py
done
