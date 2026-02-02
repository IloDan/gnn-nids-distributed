import ast
import json
import os
import pickle
import random
import time

import cudf
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cuml.ensemble import RandomForestClassifier  # cuML single-GPU
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.dataload import (  # tue utility
    Timer,
    preprocess_NUSW_dataset,
    preprocess_TON_dataset,
)
from src.utils import get_qfs_target_feature_k_rows


# ────────────────────────────────────────────────
def get_gpu_details():
    dev   = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    return {"id": dev.id, "name": props["name"]}

# ────────────────────────────────────────────────
def train_single_model(X_tr, y_tr, X_val, params, idx):
    """Addestra un RandomForest e lo salva su disco (facoltativo)."""
    model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                   max_depth   =params['max_depth'],
                                   max_features=params['max_features'],
                                   random_state=idx,
                                   n_streams   =1)
    model.fit(X_tr, y_tr)

    # opzionale: persistenza modello + hyper-params
    os.makedirs("models", exist_ok=True)
    with open(f"models/rf_model_{idx}.pkl", "wb") as f:  pickle.dump(model, f)
    with open(f"models/rf_model_{idx}_hyper.json", "w") as f: json.dump(params, f)

    return model

# ────────────────────────────────────────────────
def majority_vote(models, X_te):
    """Predizioni ensemble tramite majority voting (GPU)."""
    n_models, n_rows = len(models), X_te.shape[0]
    all_preds = cp.empty((n_models, n_rows), dtype=cp.int32)

    for i, m in enumerate(models):
        preds = m.predict(X_te)
        preds = preds.to_cupy() if hasattr(preds, "to_cupy") else cp.asarray(preds)
        all_preds[i] = preds

    # bin-count per colonna → classe con occorrenze massime
    final_preds = cp.apply_along_axis(lambda col: cp.bincount(col).argmax(),
                                      axis=0, arr=all_preds)
    return final_preds

# ────────────────────────────────────────────────
if __name__ == "__main__":
    cp.cuda.Device(0).use()                    # forziamo GPU 0
    print("GPU in uso:", get_gpu_details())

    NUM_MODELS = 10                            # dimensione ensemble
    HYP_SPACE  = {                             # spazio di ricerca semplice
        'n_estimators': (100, 500),
        'max_depth'   : (10 ,  20),
        'max_features': (0.6, 0.9)
    }

    datasets = {                               # stessi CSV di prima
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
    #     'full': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_full.csv'),
    #     'dos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_dos.csv'),
    #     'ddos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ddos.csv'),
    #     'backdoor': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_backdoor.csv'),
    #     'injection': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_injection.csv'),
    #     'password': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_password.csv'),
    #     'ransomware': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ransomware.csv'),
    #     'scanning': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_scanning.csv'),
    #     'xss': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_xss.csv'),
    # }
    
    all_qfs_features = pd.read_csv('./data/QFS_features.csv', index_col=0)
    selection_algorithms = ['QUBOCorrelation', 'QUBOMutualInformation', 'QUBOSVCBoosting']
    QUBO_solvers = ['SimulatedAnnealing', 'SteepestDescent', 'TabuSampler']

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    results_file = "results/rc_gpu_metrics_NUSW.csv"
    try:
        results_df = pd.read_csv(results_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=[
            "attack_cat", "qfs_index", "selection_algorithm_name", "QUBO_solver",
            "target_feature_k", "device", "graph_time", "train_time", "test_time",
            "accuracy", "bal_accuracy", "f1_malicious", "precision_malicious", "recall_malicious"
        ])
    
    # ------------------------------------------------------------------
    for name, df in datasets.items():
        print(f"\n=== Dataset: {name} ===")
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

                # feature_ks = sorted(selected_features_df['target_feature_k'].unique())
                # for i in range(len(selected_features_df), step=1):
                #     selected_features_row = selected_features_df.iloc[i]
                for selected_features_row in (row.iloc[0] for row in target_feature_ks):
                    qfs_index = int(selected_features_row.name)
                    feature_k = int(selected_features_row['target_feature_k'])
                
                    if not results_df[
                        (results_df['attack_cat'] == name) & 
                        (results_df['selection_algorithm_name'] == selection_algorithm) & 
                        (results_df['QUBO_solver'] == QUBO_solver) &
                        (results_df['target_feature_k'] == feature_k)
                        ].empty:
                        print(f"Skipping k={feature_k} (already computed).")
                        continue

                    print(f"\n--- Target Feature k: {feature_k} (QFS Index: {qfs_index}) ---")
                    
                    selected_features = ast.literal_eval(selected_features_row['selected_features'])
        
                    # ── preprocessing + split
                    X_tr, y_tr, X_te, y_te = preprocess_NUSW_dataset(df, scaler_type='standard', qfs_features=selected_features)
                    # X_tr, y_tr, X_te, y_te = preprocess_TON_dataset(df, scaler_type='standard')


                    X_tr = cudf.from_pandas(X_tr.astype(np.float32))
                    X_te = cudf.from_pandas(X_te.astype(np.float32))
                    y_tr = cudf.Series(y_tr.astype(np.int32))
                    y_te_np = y_te.astype(np.int32).to_numpy()          # sklearn richiede NumPy

                    # ── training ensemble
                    models = []
                    with Timer() as train_t:
                        for idx in range(NUM_MODELS):
                            params = {
                                'n_estimators': random.randint(*HYP_SPACE['n_estimators']),
                                'max_depth'   : random.randint(*HYP_SPACE['max_depth']),
                                'max_features': random.uniform(*HYP_SPACE['max_features'])
                            }
                            print(f"  Model {idx}: {params}")
                            models.append(train_single_model(X_tr, y_tr, X_te, params, idx))
                    # ── majority-voting
                    with Timer() as test_t:
                        final_preds_cp = majority_vote(models, X_te)

                    final_preds = cp.asnumpy(final_preds_cp)
                    # ── metriche
                    acc        = accuracy_score          (y_te_np, final_preds)
                    bal_acc    = balanced_accuracy_score(y_te_np, final_preds)
                    f1_mal     = f1_score               (y_te_np, final_preds, pos_label=1)
                    prec_mal   = precision_score        (y_te_np, final_preds, pos_label=1)
                    rec_mal    = recall_score           (y_te_np, final_preds, pos_label=1)

                    cm = confusion_matrix(y_te_np, final_preds)
                    plt.figure(figsize=(8,6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Benign','Attack'],
                                yticklabels=['Benign','Attack'])
                    plt.title(f'Confusion Matrix RC - {name} - QFS {qfs_index:05d}')
                    plt.xlabel('Predicted'); plt.ylabel('True')
                    plt.tight_layout()
                    plt.savefig(f'figures/singlegpu_confusion_matrix_RC_{name}_QFS_{qfs_index:05d}_NUSW.png')
                    # plt.savefig(f'singlegpu_confusion_matrix_RC_{name}_ToN.png')
                    plt.close()

                    metrics_df = pd.DataFrame([{
                        "attack_cat": name,
                        "qfs_index": qfs_index,
                        "selection_algorithm_name": selection_algorithm,
                        "QUBO_solver": QUBO_solver,
                        "target_feature_k": feature_k,
                        "device": "cuda",
                        "graph_time": 0,
                        "train_time": round(train_t.elapsed, 2),
                        "test_time": round(test_t.elapsed, 4),
                        "accuracy": round(acc, 5),
                        "bal_accuracy": round(bal_acc, 5),
                        "f1_malicious": round(f1_mal, 5),
                        "precision_malicious": round(prec_mal, 5),
                        "recall_malicious": round(rec_mal, 5)
                    }])
                    
                    results_df = pd.concat([results_df, metrics_df], ignore_index=True)

                    # pulizia memoria GPU
                    for m in models: del m
                    cp.get_default_memory_pool().free_all_blocks()

                results_df.to_csv(results_file, index=False)
                # results_df.to_csv("results/rc_gpu_metrics_ToN.csv", index=False)
                print(f"\nMetriche salvate in {results_file}")
