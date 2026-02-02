import ast
import os, json, pickle, random, pathlib
import cupy as cp
from src.utils import get_qfs_target_feature_k_rows
import cudf, pandas as pd, numpy as np
import torch.multiprocessing as mp
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix)

from src.dataload import preprocess_TON_dataset, preprocess_NUSW_dataset, Timer
import seaborn as sns
import matplotlib.pyplot as plt


# ───────────────────────── Configurazione base ─────────────────────────────
NUM_MODELS       = 10
MODELS_PER_GPU   = 5

SAVE_DIR         = pathlib.Path("models"); SAVE_DIR.mkdir(exist_ok=True)

DATA_DIR = "data/NUSW_NB15/attack_cat_splitted"
DATASETS = {
    'full': 'UNSW-NB15_splitted.csv',
    'dos': 'UNSW-NB15_dos.csv',
    'fuzzers': 'UNSW-NB15_fuzzers.csv',
    'exploits': 'UNSW-NB15_exploits.csv',
    'generic': 'UNSW-NB15_generic.csv',
    'reconnaissance': 'UNSW-NB15_reconnaissance.csv',
    'analysis': 'UNSW-NB15_analysis.csv',
    'shellcode': 'UNSW-NB15_shellcode.csv',
    'backdoor': 'UNSW-NB15_backdoor.csv',
}


def sample_hparams(rng: np.random.Generator):
    return {
        'n_estimators': int(rng.integers(100, 500)),
        'max_depth':    int(rng.integers(10, 20)),
        'max_features': float(rng.uniform(0.6, 0.9)),
        'n_streams':    4
    }


def train_on_gpu(gpu_id: int, idx_list: list[int], X_tr_np, y_tr_np, X_te_np):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cp.cuda.Device(0).use()

    X_tr = cudf.from_pandas(pd.DataFrame(X_tr_np.astype(np.float32)))
    y_tr = cudf.Series(y_tr_np.astype(np.int32))
    X_te = cudf.from_pandas(pd.DataFrame(X_te_np.astype(np.float32)))

    rng = np.random.default_rng(seed=gpu_id)

    for idx in idx_list:
        params = sample_hparams(rng)
        model  = RandomForestClassifier(**params, random_state=idx)
        model.fit(X_tr, y_tr)

        with open(SAVE_DIR / f"rf_gpu{gpu_id}_model{idx}.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(SAVE_DIR / f"rf_gpu{gpu_id}_model{idx}_hyper.json", "w") as f:
            json.dump(params, f)

    cp.get_default_memory_pool().free_all_blocks()


def majority_vote(model_files: list[pathlib.Path], X_te_np):
    cp.cuda.Device(0).use()
    X_te = cudf.from_pandas(pd.DataFrame(X_te_np.astype(np.float32)))
    n_models, n_rows = len(model_files), X_te.shape[0]
    votes = cp.empty((n_models, n_rows), dtype=cp.int32)

    for i, mfile in enumerate(model_files):
        with open(mfile, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X_te).to_cupy()
        votes[i] = preds

    final = cp.apply_along_axis(lambda col: cp.bincount(col).argmax(), axis=0, arr=votes)
    return final


def _clear_saved_models():
    for p in SAVE_DIR.glob("rf_gpu*_model*.pkl"):
        p.unlink(missing_ok=True)
    for p in SAVE_DIR.glob("rf_gpu*_model*_hyper.json"):
        p.unlink(missing_ok=True)


def process_dataset(name: str, csv_file: str):
    print(f"\n===== Dataset: {name} =====")

    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))

    all_qfs_features = pd.read_csv('./data/QFS_features.csv', index_col=0)
    selection_algorithms = ['QUBOCorrelation', 'QUBOMutualInformation', 'QUBOSVCBoosting']
    QUBO_solvers = ['SimulatedAnnealing', 'SteepestDescent', 'TabuSampler']

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    rows = []

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
                X_tr, y_tr, X_te, y_te = preprocess_NUSW_dataset(
                    df, scaler_type="standard", qfs_features=selected_features
                )
                # X_tr, y_tr, X_te, y_te = preprocess_TON_dataset(df, scaler_type="standard")

                X_tr_np, y_tr_np = X_tr.to_numpy(np.float32), y_tr.to_numpy(np.int32)
                X_te_np, y_te_np = X_te.to_numpy(np.float32), y_te.to_numpy(np.int32)

                num_gpus = cp.cuda.runtime.getDeviceCount()
                assert NUM_MODELS % num_gpus == 0, "NUM_MODELS deve essere multiplo del numero di GPU"

                model_indices = list(range(NUM_MODELS))
                shards = [model_indices[i::num_gpus] for i in range(num_gpus)]

                # IMPORTANT: pulisci modelli vecchi prima di allenare questa config
                _clear_saved_models()

                with Timer() as t_train:
                    procs: list[mp.Process] = []
                    for gpu_id, idx_list in enumerate(shards):
                        p = mp.Process(
                            target=train_on_gpu,
                            args=(gpu_id, idx_list, X_tr_np, y_tr_np, X_te_np)
                        )
                        p.start()
                        procs.append(p)
                    for p in procs:
                        p.join()

                with Timer() as t_test:
                    model_files = sorted(SAVE_DIR.glob("rf_gpu*_model*.pkl"))
                    if len(model_files) != NUM_MODELS:
                        raise RuntimeError(f"Attesi {NUM_MODELS} modelli, trovati {len(model_files)}: {model_files[:3]} ...")
                    preds_cp = majority_vote(model_files, X_te_np)
                preds = cp.asnumpy(preds_cp)

                acc     = accuracy_score(y_te_np, preds)
                bal_acc = balanced_accuracy_score(y_te_np, preds)
                f1_mal  = f1_score(y_te_np, preds, pos_label=1)
                prec    = precision_score(y_te_np, preds, pos_label=1)
                rec     = recall_score(y_te_np, preds, pos_label=1)

                cm = confusion_matrix(y_te_np, preds)

                rows.append({
                    "attack_cat": name,
                    "qfs_index": qfs_index,
                    "selection_algorithm_name": selection_algorithm,
                    "QUBO_solver": QUBO_solver,
                    "target_feature_k": feature_k,
                    "device": f"{num_gpus}×GPU",
                    "graph_time": 0,
                    "train_time": round(t_train.elapsed, 2),
                    "test_time":  round(t_test.elapsed, 4),
                    "accuracy":  round(acc, 5),
                    "bal_accuracy": round(bal_acc, 5),
                    "f1_malicious": round(f1_mal, 5),
                    "precision_malicious": round(prec, 5),
                    "recall_malicious": round(rec, 5),
                })

                # se vuoi salvare le confusion matrix per ogni config, fallo qui (opzionale)
                # plt.figure(figsize=(8, 6))
                # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                #             xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
                # plt.title(f'CM RC - {name} - {selection_algorithm} - {QUBO_solver} - k={feature_k} - QFS={qfs_index:05d}')
                # plt.savefig(f'figures/cm_rc_{name}_{selection_algorithm}_{QUBO_solver}_k{feature_k}_QFS{qfs_index:05d}.png')
                # plt.close()

    return rows


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    results = []
    for name, csv_name in DATASETS.items():
        rows = process_dataset(name, csv_name)
        results.extend(rows)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/rc_metrics_NUSW.csv", index=False)
    print("\nMetriche salvate in results/rc_metrics_NUSW.csv")
