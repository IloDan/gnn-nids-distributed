import os, json, pickle, random, pathlib
import cupy as cp
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
NUM_MODELS       = 10      # totale modelli nell' ensemble
MODELS_PER_GPU   = 5       # quanti per ogni GPU (NUM_MODELS deve essere multiplo)

SAVE_DIR         = pathlib.Path("models"); SAVE_DIR.mkdir(exist_ok=True)

# DATA_DIR         = "data/ToN_IoT/attack_cat"  # cartella dei csv
# DATASETS = {
#     'full':        'ToN_IoT_full.csv',
#     'dos':         'ToN_IoT_dos.csv',
#     'ddos':        'ToN_IoT_ddos.csv',
#     'backdoor':    'ToN_IoT_backdoor.csv',
#     'injection':   'ToN_IoT_injection.csv',
#     'password':    'ToN_IoT_password.csv',
#     'ransomware':  'ToN_IoT_ransomware.csv',
#     'scanning':    'ToN_IoT_scanning.csv',
#     'xss':         'ToN_IoT_xss.csv',
# }

DATA_DIR = "data/NUSW_NB15/attack_cat_splitted"  # cartella dei csv

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

# ─────────────────────── utilities ─────────────────────────────────────────
def sample_hparams(rng: np.random.Generator):
    """Estrae iper-parametri a caso."""
    return {
        'n_estimators': int(rng.integers(100, 500)),
        'max_depth':    int(rng.integers(10, 20)),
        'max_features': float(rng.uniform(0.6, 0.9)),
        'n_streams':    4
    }

def train_on_gpu(gpu_id: int, idx_list: list[int], X_tr_np, y_tr_np, X_te_np):
    """Processo worker: addestra tutti i modelli elencati in idx_list."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # isola il device
    cp.cuda.Device(0).use()                             # index locale

    # Conversione dati in GPU una sola volta
    X_tr = cudf.from_pandas(pd.DataFrame(X_tr_np.astype(np.float32)))
    y_tr = cudf.Series(y_tr_np.astype(np.int32))
    X_te = cudf.from_pandas(pd.DataFrame(X_te_np.astype(np.float32)))

    rng = np.random.default_rng(seed=gpu_id)

    for idx in idx_list:
        params = sample_hparams(rng)
        model  = RandomForestClassifier(**params, random_state=idx)
        model.fit(X_tr, y_tr)

        # Salvataggio modello + hyper-params (una volta finito il fit)
        with open(SAVE_DIR / f"rf_gpu{gpu_id}_model{idx}.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(SAVE_DIR / f"rf_gpu{gpu_id}_model{idx}_hyper.json", "w") as f:
            json.dump(params, f)

    # pulizia
    cp.get_default_memory_pool().free_all_blocks()


def majority_vote(model_files: list[pathlib.Path], X_te_np):
    """Calcola majority-voting in GPU."""
    cp.cuda.Device(0).use()
    X_te = cudf.from_pandas(pd.DataFrame(X_te_np.astype(np.float32)))
    n_models, n_rows = len(model_files), X_te.shape[0]
    votes = cp.empty((n_models, n_rows), dtype=cp.int32)

    for i, mfile in enumerate(model_files):
        with open(mfile, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X_te).to_cupy()
        votes[i] = preds

    # per colonna prendo la classe con occorrenze massime
    final = cp.apply_along_axis(lambda col: cp.bincount(col).argmax(), axis=0, arr=votes)
    return final


# ──────────────────────────── main ─────────────────────────────────────────

def process_dataset(name: str, csv_file: str):
    print(f"\n===== Dataset: {name} =====")

    # load + split (CPU)
    df      = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    # X_tr, y_tr, X_te, y_te = preprocess_TON_dataset(df, scaler_type="standard")
    X_tr, y_tr, X_te, y_te = preprocess_NUSW_dataset(df, scaler_type="standard")

    X_tr_np, y_tr_np = X_tr.to_numpy(np.float32), y_tr.to_numpy(np.int32)
    X_te_np, y_te_np = X_te.to_numpy(np.float32), y_te.to_numpy(np.int32)

    num_gpus = cp.cuda.runtime.getDeviceCount()
    assert NUM_MODELS % num_gpus == 0, "NUM_MODELS deve essere multiplo del numero di GPU"  # noqa

    model_indices = list(range(NUM_MODELS))
    shards = [model_indices[i::num_gpus] for i in range(num_gpus)]  # round-robin

    # ── training parallelo ────────────────────────────────────────────────
    with Timer() as t_train:
        procs: list[mp.Process] = []
        for gpu_id, idx_list in enumerate(shards):
            p = mp.Process(target=train_on_gpu,
                           args=(gpu_id, idx_list, X_tr_np, y_tr_np, X_te_np))
            p.start(); procs.append(p)
        for p in procs: p.join()

    # ── ensemble test ─────────────────────────────────────────────────────
    with Timer() as t_test:
        model_files = sorted(SAVE_DIR.glob("rf_gpu*_model*.pkl"))[-NUM_MODELS:]
        preds_cp    = majority_vote(model_files, X_te_np)
    preds = cp.asnumpy(preds_cp)

    # metriche (CPU)
    acc     = accuracy_score          (y_te_np, preds)
    bal_acc = balanced_accuracy_score(y_te_np, preds)
    f1_mal  = f1_score               (y_te_np, preds, pos_label=1)
    prec    = precision_score        (y_te_np, preds, pos_label=1)
    recall  = recall_score           (y_te_np, preds, pos_label=1)

    cm = confusion_matrix(y_te_np, preds)

    return {
        "attack_cat": name,
        "device": f"{num_gpus}×GPU",
        "graph_time": 0,
        "train_time": round(t_train.elapsed, 2),
        "test_time":  round(t_test.elapsed, 4),
        "accuracy":  round(acc, 5),
        "bal_Accuracy": round(bal_acc, 5),
        "f1_malicious": round(f1_mal, 5),
        "precision_malicious": round(prec, 5),
        "recall_malicious": round(recall, 5),
    }, cm


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    results = []

    for name, csv_name in DATASETS.items():
        res, cm  = process_dataset(name, csv_name)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        # plt.xlabel('Predicted labels')
        # plt.ylabel('True labels')
        # plt.title('Confusion Matrix Random_Committee')
        # plt.savefig(f'confusion_matrix_RC_{name}_ToN.png')
        results.append(res)

    pd.DataFrame(results).to_csv("rc_metrics_NUSW.csv", index=False)
    print("\nMetriche salvate in rc_metrics_NUSW.csv")
