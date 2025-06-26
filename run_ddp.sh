#!/bin/bash

SEEDS=(42 123 999 2025 7)
PROTOCOLS=("ToNpart" "ToN" "all" "part")
NUM_GPUS=2
PORT=12355

source ~/miniconda3/etc/profile.d/conda.sh
conda activate KC

for PROTOCOL in "${PROTOCOLS[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        # echo "🚀 Avvio esecuzione con seed $SEED e protocollo $PROTOCOL"
        # PROTOCOLS=$PROTOCOL SEED=$SEED python GNN/train_gnn.py

        echo "🚀 Avvio esecuzione con seed $SEED e protocollo $PROTOCOL DDP"
        PROTOCOLS=$PROTOCOL SEED=$SEED torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT GNN/train_ddp.py
    done
done