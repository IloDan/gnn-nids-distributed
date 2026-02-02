# ────────────────────────── logistic_regression_single_gpu.py ──────────────────────────
import ast
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset
from src.utils import get_qfs_target_feature_k_rows
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


# ⏱️ Timer context manager ----------------------------------------------------
class Timer:
    def __enter__(self):  self._t0 = time.time(); return self
    def __exit__(self, *exc): self.elapsed = time.time() - self._t0

# 🧠 Semplice regressione logistica -------------------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)              # Wx + b
    def forward(self, x):
        return self.linear(x)

# 📦 Gestore dei tensori -------------------------------------------------------
class DataHandler:
    def __init__(self, Xtr, ytr, Xte, yte, device):
        self.Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
        self.ytr = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(1)
        self.Xte = torch.tensor(Xte, dtype=torch.float32, device=device)
        self.yte = torch.tensor(yte, dtype=torch.float32, device=device).unsqueeze(1)
    def train_data(self): return self.Xtr, self.ytr
    def test_data (self): return self.Xte, self.yte

# 🏋️‍♂️ Training + evaluation --------------------------------------------------
class ModelTrainer:
    def __init__(self, model, data, crit, opt, sch, epochs):
        self.m, self.d, self.crit, self.opt, self.sch, self.e = model, data, crit, opt, sch, epochs
        self.train_timer = None
        
    def train(self):
        with Timer() as self.train_timer:
            for _ in tqdm(range(self.e), desc="Training", leave=False):
                self.m.train()
                X, y = self.d.train_data()
                self.opt.zero_grad()
                loss = self.crit(self.m(X), y)
                loss.backward()
                self.opt.step()
                self.sch.step()
    @torch.no_grad()
    def evaluate(self):
        self.m.eval()
        X, y = self.d.test_data()
        with Timer() as test_t:
            preds = (torch.sigmoid(self.m(X)) > .5).float()
        acc = (preds == y).float().mean().item()

        y_true = y.cpu().numpy().astype(int)
        y_pred = preds.cpu().numpy().astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score   (y_true, y_pred, pos_label=1, zero_division=0)
        f1   = f1_score      (y_true, y_pred, pos_label=1, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred)
        return acc, prec, rec, f1, cm, test_t.elapsed, bal_acc

# 🧪 Main ----------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    datasets = {                                       # stessi CSV di prima
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

    results = []
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    results_file = "results/metrics_logreg_single_gpu_NUSW.csv"
    try:
        results_df = pd.read_csv(results_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=[
            "attack_cat", "qfs_index", "selection_algorithm_name", "QUBO_solver",
            "target_feature_k", "device", "graph_time", "train_time", "test_time",
            "accuracy", "bal_accuracy", "f1_malicious", "precision_malicious", "recall_malicious"
        ])

    for name, df in datasets.items():
        print(f"\n📂 Dataset: {name}")
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
                        print(f"⚠️  Skipping QFS index {qfs_index:05d} (k={feature_k}) - already computed.")
                        continue
                    
                    print(f"\n➡️  Training with QFS index {qfs_index:05d} (k={feature_k})...")
                    
                    selected_features = ast.literal_eval(selected_features_row['selected_features'])
                    
                    Xtr, ytr, Xte, yte = preprocess_NUSW_dataset(df, scaler_type='standard', qfs_features=selected_features)
                    # Xtr, ytr, Xte, yte = preprocess_TON_dataset(df, scaler_type='standard')
                    Xtr, Xte = Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32)
                    ytr, yte = ytr.to_numpy(np.int32),   yte.to_numpy(np.int32)

                    data = DataHandler(Xtr, ytr, Xte, yte, device)
                    model = LogisticRegressionModel(input_dim=Xtr.shape[1]).to(device)

                    optim_   = optim.SGD(model.parameters(), lr=1.5, weight_decay=1e-5)
                    sched_   = LambdaLR(optim_, lambda ep: 1 / (1 + ep*1e-5))
                    trainer  = ModelTrainer(model, data, nn.BCEWithLogitsLoss(),
                                            optim_, sched_, epochs=1500)

                    trainer.train()
                    acc, prec, rec, f1, cm, test_t, bal_acc = trainer.evaluate()

                    # confusion matrix → immagine
                    plt.figure(figsize=(6,5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Benign','Attack'],
                                yticklabels=['Benign','Attack'])
                    plt.title(f'Confusion Matrix – {name} - QFS {qfs_index:05d}')
                    plt.savefig(f'figures/singlegpu_sgd_cm_{name}_QFS_{qfs_index:05d}_NUSW.png')
                    plt.close()
                    # plt.savefig(f'figures/singlegpu_sgd_cm_{name}_ToN.png'); plt.close()

                    metrics_df = pd.DataFrame([{
                        "attack_cat": name,
                        "qfs_index": qfs_index,
                        "selection_algorithm_name": selection_algorithm,
                        "QUBO_solver": QUBO_solver,
                        "target_feature_k": feature_k,
                        "device": str(device),
                        "graph_time": 0,
                        "train_time": round(trainer.train_timer.elapsed, 2),
                        "test_time" : round(test_t, 4),
                        "accuracy": round(acc, 5),
                        "bal_accuracy": round(bal_acc, 5),
                        "f1_malicious": round(f1, 5),
                        "precision_malicious": round(prec, 5),
                        "recall_malicious": round(rec, 5)
                    }])

                    results_df = pd.concat([results_df, metrics_df], ignore_index=True)

                results_df.to_csv(results_file, index=False)
                print(f"\n✅  Metrics saved to {results_file}")
                # pd.DataFrame(results).to_csv("results/metrics_logreg_single_gpu_ToN.csv", index=False)
                # print("\n✅  Metrics saved to 'results/sgd_gpu_metrics_ToN.csv")
