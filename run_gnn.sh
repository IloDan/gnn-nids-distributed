#!/bin/bash

SEEDS=(42)
PROTOCOLS=("all" "part")

source ~/miniconda3/etc/profile.d/conda.sh
conda activate KC

for PROTOCOL in "${PROTOCOLS[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        echo "🚀 Avvio esecuzione con seed $SEED e protocollo $PROTOCOL"
        SEED=$SEED PROTOCOLS=$PROTOCOL python GNN/train_gnn.py
    done
done