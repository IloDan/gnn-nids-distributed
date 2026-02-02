# distributed_log_reg.py
"""Distributed logistic-regression benchmark on ToN-IoT / UNSW-NB15 datasets.

Launch with torchrun, e.g.:

  torchrun --standalone --nnodes 1 --nproc_per_node 4 distributed_log_reg.py \
      --dataset full --epochs 1500 --lr 1.5 --data_root data/NUSW_NB15/attack_cat

Every process handles its own shard of the data thanks to ``DistributedSampler``;
metrics are averaged across GPUs with ``torch.distributed.all_reduce`` and only
rank‑0 writes the final CSV.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from src.utils import get_qfs_target_feature_k_rows
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# 🏷️ Import project‑local preprocessing helpers
from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset

# ────────────────────────────────────────────────────────────────────────────────
#                                Utilities
# ────────────────────────────────────────────────────────────────────────────────

def setup_distributed() -> Tuple[int, int, torch.device]:
    """Initialise *torch.distributed* from environment variables set by torchrun.

    Returns
    -------
    rank, world_size, device
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    return rank, world, device


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a scalar *torch.Tensor* across processes in‑place."""
    rt = t.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


# ────────────────────────────────────────────────────────────────────────────────
#                                Model
# ────────────────────────────────────────────────────────────────────────────────

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.linear(x)


# ────────────────────────────────────────────────────────────────────────────────
#                               Training
# ────────────────────────────────────────────────────────────────────────────────

def train_one_dataset(
    name: str,
    df: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    world: int,
) -> list[Dict[str, float]]:
    print(f"\n=== Dataset: {name} ===")
    
    all_qfs_features = pd.read_csv('./data/QFS_features.csv', index_col=0)
    selection_algorithms = ['QUBOCorrelation', 'QUBOMutualInformation', 'QUBOSVCBoosting']
    QUBO_solvers = ['SimulatedAnnealing', 'SteepestDescent', 'TabuSampler']
    
    os.makedirs("figures", exist_ok=True)

    results = []   # <── lista di metriche solo per rank 0
    
    for selection_algorithm in selection_algorithms:
        print(f"\n=== Selection Algorithm: {selection_algorithm} ===")
        for QUBO_solver in QUBO_solvers:
            print(f"\n=== QUBO Solver: {QUBO_solver} ===")                
            selected_features_df = all_qfs_features[
                (all_qfs_features['category'] == name) &
                (all_qfs_features['selection_algorithm_name'] == selection_algorithm) &
                (all_qfs_features['QUBO_solver'] == QUBO_solver)
            ].sort_values(by='target_feature_k')
            
            target_feature_ks = get_qfs_target_feature_k_rows(selected_features_df)
            
            for selected_features_row in (row.iloc[0] for row in target_feature_ks):
                qfs_index = int(selected_features_row.name)
                feature_k = int(selected_features_row['target_feature_k'])
                selected_features = ast.literal_eval(selected_features_row['selected_features'])
                            
                # preprocess
                if args.ton:
                    X_train, y_train, X_test, y_test = preprocess_TON_dataset(
                        df, scaler_type="standard"
                    )
                else:
                    X_train, y_train, X_test, y_test = preprocess_NUSW_dataset(
                        df, scaler_type="standard", qfs_features=selected_features
                    )

                X_train_np, X_test_np = X_train.values.astype(np.float32), X_test.values.astype(np.float32)
                y_train_np, y_test_np = y_train.values.astype(np.float32), y_test.values.astype(np.float32)

                train_ds = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(y_train_np))
                test_ds = TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(y_test_np))

                train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
                test_sampler = DistributedSampler(test_ds, num_replicas=world, rank=rank, shuffle=False)

                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True
                )
                test_loader = DataLoader(
                    test_ds, batch_size=args.batch_size * 4, sampler=test_sampler, pin_memory=True
                )

                model = LogisticRegressionModel(input_dim=X_train_np.shape[1]).to(device)
                model = DDP(model, device_ids=[device.index], output_device=device.index)

                optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
                scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch * 1e-5))
                criterion = nn.BCEWithLogitsLoss().to(device)

                # train
                model.train()
                for epoch in range(args.epochs):
                    train_sampler.set_epoch(epoch)
                    for xb, yb in train_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.unsqueeze(1).to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                

                    # ➏ Eval
                model.eval()
                y_true, y_pred = [], []
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)          # <-- target su GPU

                        logits = model(xb)
                        preds = (torch.sigmoid(logits) > 0.5).float()  # su GPU

                        y_true.append(yb)
                        y_pred.append(preds.squeeze())

                # concat su GPU
                y_true_t = torch.cat(y_true)   # CUDA
                y_pred_t = torch.cat(y_pred)   # CUDA

                # Gather across ranks (sempre CUDA con backend=nccl)
                y_true_g = [torch.zeros_like(y_true_t) for _ in range(world)]
                y_pred_g = [torch.zeros_like(y_pred_t) for _ in range(world)]
                torch.distributed.all_gather(y_true_g, y_true_t)
                torch.distributed.all_gather(y_pred_g, y_pred_t)

                if rank == 0:
                    # ora posso tornare su CPU per le metriche
                    y_true_np = torch.cat(y_true_g).cpu().numpy().astype(int)
                    y_pred_np = torch.cat(y_pred_g).cpu().numpy().astype(int)

                    precision = precision_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
                    recall = recall_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
                    f1 = f1_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)
                    accuracy = (y_true_np == y_pred_np).mean()

                    results.append({
                        "attack_cat": name,
                        "qfs_index": qfs_index,
                        "selection_algorithm_name": selection_algorithm,
                        "QUBO_solver": QUBO_solver,
                        "target_feature_k": feature_k,
                        "accuracy": accuracy,
                        "bal_accuracy": bal_acc,
                        "f1_malicious": f1,
                        "precision_malicious": precision,
                        "recall_malicious": recall,
                    })

    return results if rank == 0 else []



# ────────────────────────────────────────────────────────────────────────────────
#                                Main entry
# ────────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed logistic regression for NIDS datasets")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1.5)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--dataset", type=str, default="full", help="Which key in dataset dict to train on (or 'all')")
    p.add_argument("--data_root", type=Path, default=Path("data/ToN_IoT/attack_cat"))
    p.add_argument("--ton", action="store_true", help="Use ToN‑IoT preprocessing instead of UNSW‑NB15")
    return p.parse_args()


def build_dataset_dict(root: Path, ton: bool) -> Dict[str, pd.DataFrame]:
    if ton:
        return {
            "full": pd.read_csv(root / "ToN_IoT_full.csv"),
            "dos": pd.read_csv(root / "ToN_IoT_dos.csv"),
            "ddos": pd.read_csv(root / "ToN_IoT_ddos.csv"),
            "backdoor": pd.read_csv(root / "ToN_IoT_backdoor.csv"),
            "injection": pd.read_csv(root / "ToN_IoT_injection.csv"),
            "password": pd.read_csv(root / "ToN_IoT_password.csv"),
            "ransomware": pd.read_csv(root / "ToN_IoT_ransomware.csv"),
            "scanning": pd.read_csv(root / "ToN_IoT_scanning.csv"),
            "xss": pd.read_csv(root / "ToN_IoT_xss.csv"),
        }
    else:
        nb_root = Path("data/NUSW_NB15")
        return {
            "full": pd.read_csv(nb_root / "UNSW-NB15_splitted.csv"),
            "dos": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_dos.csv"),
            "fuzzers": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_fuzzers.csv"),
            "exploits": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_exploits.csv"),
            "generic": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_generic.csv"),
            "reconnaissance": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_reconnaissance.csv"),
            "analysis": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_analysis.csv"),
            "shellcode": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_shellcode.csv"),
            "backdoor": pd.read_csv(nb_root / "attack_cat_splitted/UNSW-NB15_backdoor.csv"),
        }


def main():
    args = parse_args()
    rank, world, device = setup_distributed()

    # Build dataset dict on every rank (CSV reading is local‑FS, cheaper than broadcast)
    datasets = build_dataset_dict(args.data_root, args.ton)

    # Select subset
    keys = [args.dataset] if args.dataset != "all" else list(datasets.keys())

    results = []
    for k in keys:
        if rank == 0:
            print(f"⇒ Processing dataset: {k}")
        res_list = train_one_dataset(k, datasets[k], device, args, rank, world)
        if rank == 0:
            results.extend(res_list)


    if rank == 0 and results:
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        result_df = pd.DataFrame(results)
        csv_path = out_dir / "sgd_metrics_ddp.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"✓ Metrics saved to {csv_path}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
