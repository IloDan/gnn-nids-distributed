from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# ⏱️ Timer context manager
class Timer:
    def __enter__(self):
        self.tick = time.time()
        return self
    def __exit__(self, *args, **kwargs):
        self.tock = time.time()
        self.elapsed = self.tock - self.tick

# 🧠 Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

# 📦 Data handler
class DataHandler:
    def __init__(self, X_train_np, y_train_np, X_test_np, y_test_np, device):
        self.X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
        self.X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        self.y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

    def get_train_data(self): return self.X_train, self.y_train
    def get_test_data(self): return self.X_test, self.y_test

# 🏋️‍♂️ Training wrapper
class ModelTrainer:
    def __init__(self, model, dataloader, criterion, optimizer, scheduler, num_epochs, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.t = None

    def train(self):
        with Timer() as self.t:
            for epoch in tqdm(range(self.num_epochs), desc="Training Progress", leave=False):
                self.model.train()
                self.optimizer.zero_grad()
                X_train, y_train = self.dataloader.get_train_data()
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            with Timer() as test_timer:
                X_test, y_test = self.dataloader.get_test_data()
                test_outputs = self.model(X_test)
                predicted = (torch.sigmoid(test_outputs) > 0.5).float()

            acc = (predicted == y_test).float().mean().item()
            y_true = y_test.cpu().numpy().astype(int)
            y_pred = predicted.cpu().numpy().astype(int)
            precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            bal_accuracy = balanced_accuracy_score(y_true, y_pred)

            return acc, precision, recall, f1, y_true, y_pred, test_timer.elapsed, bal_accuracy

# 🧪 Main routine
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        'full': pd.read_csv('data/NUSW_NB15/UNSW-NB15_splitted.csv'),
        'dos': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_dos.csv'),
        'fuzzers': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_fuzzers.csv'),
        'exploits': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_exploits.csv'),
        'generic': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_generic.csv'),
        'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_reconnaissance.csv'),
        'analysis': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_analysis.csv'),
        'shellcode': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_shellcode.csv'),
        'backdoor': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_backdoor.csv'),
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

    results = []

    for name, df in datasets.items():
        print(f"\n🔍 Dataset: {name}")
        X_train, y_train, X_test, y_test = preprocess_NUSW_dataset(df, scaler_type='standard')
        # X_train, y_train, X_test, y_test = preprocess_TON_dataset(df, scaler_type='standard')
        X_train_np, X_test_np = X_train.to_numpy().astype('float32'), X_test.to_numpy().astype('float32')
        y_train_np, y_test_np = y_train.to_numpy().astype('int32'), y_test.to_numpy().astype('int32')

        data_handler = DataHandler(X_train_np, y_train_np, X_test_np, y_test_np, device)
        model = LogisticRegressionModel(input_dim=X_train_np.shape[1]).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            model = nn.DataParallel(model)

        optimizer = optim.SGD(model.parameters(), lr=1.5, weight_decay=1e-5)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch * 1e-5))
        trainer = ModelTrainer(model, data_handler, nn.BCEWithLogitsLoss(), optimizer, scheduler, num_epochs=1500, device=device)

        trainer.train()
        accuracy, precision, recall, f1, y_true, y_pred, test_time, bal_acc= trainer.evaluate()

        # 📊 Save confusion matrix
        # cm = confusion_matrix(y_true, y_pred)
        # plt.figure(figsize=(6, 5))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        # plt.title(f'Confusion Matrix - {name}')
        # plt.savefig(f'cm_{name}_ToN.png')
        # plt.close()

        results.append({
            "attack_cat": name,
            "device": str(device),
            "graph_time": 0,
            "train_time": round(trainer.t.elapsed, 2),
            "test_time": round(test_time, 4),
            "accuracy": round(accuracy, 5),
            "bal_accuracy": round(bal_acc, 5),
            "f1_malicious": round(f1, 5),
            "precision_malicious": round(precision, 5),
            "recall_malicious": round(recall, 5)
        })

    # 📁 Save results to CSV
    result_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    result_df.to_csv("results/sgd_metrics_ToN_prov.csv", index=False)
    print("\n✅ Metrics saved to 'results/metrics_logistic_regression.csv'")

