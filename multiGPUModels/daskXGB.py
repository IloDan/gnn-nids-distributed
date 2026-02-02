import ast
import os
import cupy
import pandas as pd
from src.utils import get_qfs_target_feature_k_rows
import numpy as np

from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from xgboost import dask as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.dataload import load_data_dask_NUWS, load_data_dask_ToN, Timer

def train_xgb(client, dtrain):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "verbosity": 1,
    }
    num_round = 100
    return xgb.train(client, params, dtrain, num_boost_round=num_round, evals=[(dtrain, "train")])


class NoAffinityLocalCUDACluster(LocalCUDACluster):
    """LocalCUDACluster senza CPU affinity (evita os.sched_setaffinity -> Errno 22 su Slurm/cgroups)."""
    def new_worker_spec(self):
        spec = super().new_worker_spec()
        for ws in spec.values():
            plugins = ws["options"].get("plugins", set())
            ws["options"]["plugins"] = {p for p in plugins if p.__class__.__name__ != "CPUAffinity"}
        return spec


if __name__ == "__main__":

    datasets = {
        "full": pd.read_csv("data/NUSW_NB15/UNSW-NB15_splitted.csv"),
        "dos": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_dos.csv"),
        "fuzzers": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_fuzzers.csv"),
        "exploits": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_exploits.csv"),
        "generic": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_generic.csv"),
        "reconnaissance": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_reconnaissance.csv"),
        "analysis": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_analysis.csv"),
        "shellcode": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_shellcode.csv"),
        "backdoor": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_backdoor.csv"),
    }

    # ToN (se vuoi)
    # datasets = {
    #     "full": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_full.csv"),
    #     "dos": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_dos.csv"),
    #     "ddos": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_ddos.csv"),
    #     "backdoor": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_backdoor.csv"),
    #     "injection": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_injection.csv"),
    #     "password": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_password.csv"),
    #     "ransomware": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_ransomware.csv"),
    #     "scanning": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_scanning.csv"),
    #     "xss": pd.read_csv("data/ToN_IoT/attack_cat/ToN_IoT_xss.csv"),
    # }

    all_qfs_features = pd.read_csv("./data/QFS_features.csv", index_col=0)
    selection_algorithms = ["QUBOCorrelation", "QUBOMutualInformation", "QUBOSVCBoosting"]
    QUBO_solvers = ["SimulatedAnnealing", "SteepestDescent", "TabuSampler"]

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results_file = "results/xgb_dask_results_NUSW.csv"
    try:
        results_df = pd.read_csv(results_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=[
            "attack_cat", "qfs_index", "selection_algorithm_name", "QUBO_solver",
            "target_feature_k", "device", "graph_time", "train_time", "test_time",
            "accuracy", "bal_accuracy", "f1_malicious", "precision_malicious", "recall_malicious"
        ])

    for name, df in datasets.items():
        print(f"\n=== Dataset: {name} ===")
        for selection_algorithm in selection_algorithms:
            print(f"\n=== Selection Algorithm: {selection_algorithm} ===")
            for QUBO_solver in QUBO_solvers:
                print(f"\n=== QUBO Solver: {QUBO_solver} ===")

                selected_features_df = all_qfs_features[
                    (all_qfs_features["category"] == name) &
                    (all_qfs_features["selection_algorithm_name"] == selection_algorithm) &
                    (all_qfs_features["QUBO_solver"] == QUBO_solver)
                ].sort_values(by="target_feature_k")

                target_feature_ks = get_qfs_target_feature_k_rows(selected_features_df)

                for selected_features_row in (row.iloc[0] for row in target_feature_ks):
                    qfs_index = int(selected_features_row.name)
                    feature_k = int(selected_features_row["target_feature_k"])

                    if not results_df[
                        (results_df["attack_cat"] == name) &
                        (results_df["selection_algorithm_name"] == selection_algorithm) &
                        (results_df["QUBO_solver"] == QUBO_solver) &
                        (results_df["target_feature_k"] == feature_k)
                    ].empty:
                        print(f"Skipping k={feature_k} (already computed).")
                        continue

                    print(f"\n--- Target Feature k: {feature_k} (QFS Index: {qfs_index}) ---")

                    selected_features = ast.literal_eval(selected_features_row["selected_features"])

                    # === DASK-CUDA CLUSTER (2 GPU) + NO AFFINITY ===
                    with NoAffinityLocalCUDACluster(
                        n_workers=2,
                        threads_per_worker=1,
                        CUDA_VISIBLE_DEVICES="0,1",
                    ) as cluster, Client(cluster) as client:

                        # carica i dati in dask (come fai già tu)
                        X_train_dask, y_train_dask, X_test_dask, y_test_dask = load_data_dask_NUWS(
                            df, scaler_type="standard", qfs_features=selected_features
                        )
                        # X_train_dask, y_train_dask, X_test_dask, y_test_dask = load_data_dask_ToN(df, scaler_type='standard')
                        print(type(X_train_dask), type(y_train_dask))

                        dtrain = xgb.DaskDMatrix(client, X_train_dask, y_train_dask)
                        dtest = xgb.DaskDMatrix(client, X_test_dask, y_test_dask)

                        with Timer() as t:
                            bst = train_xgb(client, dtrain)
                        train_time = t.elapsed

                        with Timer() as t_test:
                            y_pred_prob = xgb.predict(client, bst["booster"], dtest).compute()
                        test_time = t_test.elapsed

                        y_pred = (y_pred_prob > 0.5).astype(int)

                        y_test_np = y_test_dask.compute()
                        if hasattr(y_test_np, "to_numpy"):
                            y_test_np = y_test_np.to_numpy()

                        accuracy = accuracy_score(y_test_np, y_pred)
                        balanced_accuracy = balanced_accuracy_score(y_test_np, y_pred)
                        f1_malicious = f1_score(y_test_np, y_pred, pos_label=1)
                        precision = precision_score(y_test_np, y_pred, pos_label=1)
                        recall = recall_score(y_test_np, y_pred, pos_label=1)

                        cm = confusion_matrix(y_test_np, y_pred)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"]
                        )
                        plt.xlabel("Predicted labels")
                        plt.ylabel("True labels")
                        plt.title(f"Confusion Matrix - {name} - QFS {qfs_index:05d}")
                        plt.savefig(f"figures/confusion_matrix_XGB_dask_{name}_QFS_{qfs_index:05d}_NUSW.png")
                        plt.close()

                        device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()

                    metrics_df = pd.DataFrame([{
                        "attack_cat": name,
                        "qfs_index": qfs_index,
                        "selection_algorithm_name": selection_algorithm,
                        "QUBO_solver": QUBO_solver,
                        "target_feature_k": feature_k,
                        "device": device,
                        "graph_time": 0,
                        "train_time": round(train_time, 2),
                        "test_time": round(test_time, 4),
                        "accuracy": round(accuracy, 5),
                        "bal_accuracy": round(balanced_accuracy, 5),
                        "f1_malicious": round(f1_malicious, 5),
                        "precision_malicious": round(precision, 5),
                        "recall_malicious": round(recall, 5),
                    }])

                    results_df = pd.concat([results_df, metrics_df], ignore_index=True)
                    results_df.to_csv(results_file, index=False)

                results_df.to_csv(results_file, index=False)
