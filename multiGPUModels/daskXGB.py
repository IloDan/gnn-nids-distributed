import os
import cupy
import asyncio
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
from xgboost import dask as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataload import load_data_dask_NUWS, load_data_dask_ToN, Timer

def train_xgb(client, dtrain):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'verbosity': 1
    }
    num_round = 100
    bst = xgb.train(client, params, dtrain, num_boost_round=num_round, evals=[(dtrain, "train")])
    return bst

if __name__ == "__main__":

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
        cluster = LocalCluster(n_workers=2, threads_per_worker=1)
        client = Client(cluster)

        X_train_dask, y_train_dask, X_test_dask, y_test_dask = load_data_dask_NUWS(df, scaler_type='standard')
        # X_train_dask, y_train_dask, X_test_dask, y_test_dask = load_data_dask_ToN(df, scaler_type='standard')
        dtrain = xgb.DaskDMatrix(client, X_train_dask, y_train_dask)
        dtest = xgb.DaskDMatrix(client, X_test_dask, y_test_dask)

        with Timer() as t:
            bst = train_xgb(client, dtrain)
        train_time = t.elapsed

        with Timer() as t_test:
            y_pred_prob = xgb.predict(client, bst['booster'], dtest).compute()
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'confusion_matrix_XGB_dask_{name}_NUSW.png')
        plt.close()

        device = cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()
        results.append({
            "attack_cat": name,
            "device": device,
            "graph_time": 0,
            "train_time": round(train_time, 2),
            "test_time": round(test_time, 4),
            "accuracy": round(accuracy, 5),
            "bal_accuracy": round(balanced_accuracy, 5),
            "f1_malicious": round(f1_malicious, 5),
            "precision_malicious": round(precision, 5),
            "recall_malicious": round(recall, 5)
        })

        client.close()
        cluster.close()

    # Salva in CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("xgb_dask_results_NUSW.csv", index=False)
