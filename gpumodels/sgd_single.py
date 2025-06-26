# ────────────────────────── logistic_regression_single_gpu.py ──────────────────────────
import os, time
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, balanced_accuracy_score)
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset

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

    # datasets = {                                       # stessi CSV di prima
    #     'full'          : pd.read_csv('data/NUSW_NB15/UNSW-NB15_splitted.csv'),
    #     'dos'           : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_dos.csv'),
    #     'fuzzers'       : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_fuzzers.csv'),
    #     'exploits'      : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_exploits.csv'),
    #     'generic'       : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_generic.csv'),
    #     'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_reconnaissance.csv'),
    #     'analysis'      : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_analysis.csv'),
    #     'shellcode'     : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_shellcode.csv'),
    #     'backdoor'      : pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_backdoor.csv'),
    # }
    
    datasets = {
        'full': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_full.csv'),
        'dos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_dos.csv'),
        'ddos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ddos.csv'),
        'backdoor': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_backdoor.csv'),
        'injection': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_injection.csv'),
        'password': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_password.csv'),
        'ransomware': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ransomware.csv'),
        'scanning': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_scanning.csv'),
        'xss': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_xss.csv'),
    }

    results = []

    for name, df in datasets.items():
        print(f"\n📂 Dataset: {name}")
        # Xtr, ytr, Xte, yte = preprocess_NUSW_dataset(df, scaler_type='standard')
        Xtr, ytr, Xte, yte = preprocess_TON_dataset(df, scaler_type='standard')
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
        plt.title(f'Confusion Matrix – {name}')
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f'figures/singlegpu_sgd_cm_{name}_ToN.png'); plt.close()

        results.append({
            "attack_cat": name,
            "device": str(device),
            "graph_time": 0,
            "train_time_s": round(trainer.train_timer.elapsed, 2),
            "test_time_s" : round(test_t, 4),
            "accuracy": round(acc, 5),
            "bal_accuracy": round(bal_acc, 5),
            "f1_malicious": round(f1, 5),
            "precision_malicious": round(prec, 5),
            "recall_malicious": round(rec, 5)
        })


    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/metrics_logreg_single_gpu_ToN.csv", index=False)
    print("\n✅  Metrics saved to 'results/sgd_gpu_metrics_ToN.csv")
