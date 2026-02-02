import ast
import os
import time

import cudf
import cupy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset
from src.utils import get_qfs_target_feature_k_rows


# ────────────────────────────────────────────────
class Timer:
    def __enter__(self):  self._t0 = time.time(); return self
    def __exit__(self, *exc): self.elapsed = time.time() - self._t0
# ────────────────────────────────────────────────
def get_gpu_details():
    dev = cupy.cuda.Device(0)
    props = cupy.cuda.runtime.getDeviceProperties(dev.id)
    return {
        "device_id": dev.id,
        "gpu_name": props["name"],
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "Not set")
    }

# ────────────────────────────────────────────────
if __name__ == "__main__":

    datasets = {
        'full'          : pd.read_csv('data/NUSW_NB15/UNSW-NB15_splitted.csv'),
        'dos'           : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_dos.csv'),
        'fuzzers'       : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_fuzzers.csv'),
        'exploits'      : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_exploits.csv'),
        'generic'       : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_generic.csv'),
        'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_reconnaissance.csv'),
        'analysis'      : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_analysis.csv'),
        'shellcode'     : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_shellcode.csv'),
        'backdoor'      : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_backdoor.csv'),
    }


    # datasets = {
    # 'full': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_full.csv'),
    # 'dos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_dos.csv'),
    # 'ddos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ddos.csv'),
    # 'backdoor': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_backdoor.csv'),
    # 'injection': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_injection.csv'),
    # 'password': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_password.csv'),
    # 'ransomware': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ransomware.csv'),
    # 'scanning': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_scanning.csv'),
    # 'xss': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_xss.csv'),
    # }
    
    all_qfs_features = pd.read_csv('./data/QFS_features.csv', index_col=0)
    selection_algorithms = ['QUBOCorrelation', 'QUBOMutualInformation', 'QUBOSVCBoosting']
    QUBO_solvers = ['SimulatedAnnealing', 'SteepestDescent', 'TabuSampler']
    
    gpu_info = get_gpu_details()
    print("Running on:", gpu_info)

    all_metrics = []
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    results_file = "results/rf_gpu_metrics_NUSW.csv"
    try:
        results_df = pd.read_csv(results_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=[
            "attack_cat", "qfs_index", "selection_algorithm_name", "QUBO_solver",
            "target_feature_k", "device", "graph_time", "train_time", "test_time",
            "accuracy", "bal_accuracy", "f1_malicious", "precision_malicious", "recall_malicious"
        ])

    # -------------------------------------------------
    for name, df in datasets.items():
        print(f"\n==================== Attack Category: {name} ====================")
        for selection_algorithm in selection_algorithms:
            print(f"\n📂 Selection Algorithm: {selection_algorithm}")
            for QUBO_solver in QUBO_solvers:
                print(f"\n📂 QUBO Solver: {QUBO_solver}")
                selected_features_df = all_qfs_features[
                    (all_qfs_features['category'] == name) &
                    (all_qfs_features['selection_algorithm_name'] == selection_algorithm) &
                    (all_qfs_features['QUBO_solver'] == QUBO_solver)
                ].sort_values(by='target_feature_k')
                
                target_feature_ks = get_qfs_target_feature_k_rows(selected_features_df)
                
                # feature_ks = sorted(selected_features_df['target_feature_k'].unique())
                for selected_features_row in (row.iloc[0] for row in target_feature_ks):
                    qfs_index = int(selected_features_row.name)
                    feature_k = int(selected_features_row['target_feature_k'])
                
                    if not results_df[
                        (results_df['attack_cat'] == name) & 
                        (results_df['selection_algorithm_name'] == selection_algorithm) & 
                        (results_df['QUBO_solver'] == QUBO_solver) &
                        (results_df['target_feature_k'] == feature_k)
                        ].empty:
                        print(f"⚠️  Skipping QFS index {qfs_index:05d} (k={feature_k}) - already computed.")
                        continue
                    
                    print(f"\n➡️  Training with QFS index {qfs_index:05d} (k={feature_k})...")
                    
                    selected_features = ast.literal_eval(selected_features_row['selected_features'])
                    
                    # ⇢ pre-processing (scaling + train/test split, già definito in src.dataload)
                    X_tr, y_tr, X_te, y_te = preprocess_NUSW_dataset(df, scaler_type='standard', qfs_features=selected_features)
                    # X_tr, y_tr, X_te, y_te = preprocess_TON_dataset(df, scaler_type='standard')

                    # ⇢ conversione in cuDF (float32 / int32)
                    X_tr = cudf.from_pandas(X_tr.astype(np.float32))
                    X_te = cudf.from_pandas(X_te.astype(np.float32))
                    y_tr = cudf.Series(y_tr.astype(np.int32))
                    y_te_np = y_te.astype(np.int32).to_numpy()          # sklearn metrics richiede NumPy

                    # ⇢ modello Random Forest (cuML single-GPU)
                    rf = RandomForestClassifier(n_estimators=100,
                                                n_streams=1,
                                                random_state=0)

                    with Timer() as tr_t:
                        rf.fit(X_tr, y_tr)

                    with Timer() as te_t:
                        y_pred_cudf = rf.predict(X_te)
                    y_pred_np = cupy.asnumpy(y_pred_cudf.values)

                    # ⇢ metriche (cuML + sklearn)
                    acc         = float(accuracy_score(y_te_np, y_pred_np))
                    bal_acc     = float(balanced_accuracy_score(y_te_np, y_pred_np))
                    f1_mal      = float(f1_score(y_te_np, y_pred_np, pos_label=1))
                    prec_mal    = float(precision_score(y_te_np, y_pred_np, pos_label=1))
                    rec_mal     = float(recall_score(y_te_np, y_pred_np, pos_label=1))

                    # ⇢ confusion matrix & plot
                    cm_cu   = confusion_matrix(y_te_np, y_pred_np)
                    cm      = cupy.asnumpy(cm_cu)
                    # plt.figure(figsize=(8,6))
                    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    #             xticklabels=['Benign','Attack'],
                    #             yticklabels=['Benign','Attack'])
                    # plt.title(f'Confusion Matrix RF - {name} - QFS {qfs_index:05d}')
                    # plt.xlabel('Predicted'); plt.ylabel('True')
                    # plt.tight_layout()
                    # plt.savefig(f'figures/singlegpu_confusion_matrix_RF_{name}_QFS_{qfs_index:05d}_NUSW.png')
                    # plt.close()

                    metrics_df = pd.DataFrame([{
                        "attack_cat": name,
                        "qfs_index": qfs_index,
                        "selection_algorithm_name": selection_algorithm,
                        "QUBO_solver": QUBO_solver,
                        "target_feature_k": feature_k,
                        "device": "cuda",
                        "graph_time": 0,
                        "train_time": round(tr_t.elapsed, 2),
                        "test_time": round(te_t.elapsed, 4),
                        "accuracy": round(acc, 5),
                        "bal_accuracy": round(bal_acc, 5),
                        "f1_malicious": round(f1_mal, 5),
                        "precision_malicious": round(prec_mal, 5),
                        "recall_malicious": round(rec_mal, 5)
                    }])

                    results_df = pd.concat([results_df, metrics_df], ignore_index=True)

                    # liberiamo memoria GPU
                    del rf, X_tr, X_te, y_tr, y_pred_cudf
                    cupy.get_default_memory_pool().free_all_blocks()

                results_df.to_csv(results_file, index=False)
                # results_df.to_csv("results/rf_gpu_metrics_ToN.csv", index=False)
                print(f"Metriche salvate in {results_file}")
