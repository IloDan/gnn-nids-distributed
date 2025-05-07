import os
import timeit
import torch as th
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from dataload import preprocess_NUSW_dataset, preprocess_ToN_dataset, preprocess_NUSW_dataset_optimized
from EGraphSAGE import EGraphSAGE, compute_accuracy

def set_seed(seed):
    import random
    import numpy as np
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def train_egraphsage(G, num_epochs=200, lr=1e-3, device='cpu', verbose=True, seed=42):
    set_seed(seed)
    device = th.device(device)
    G = G.to(device)

    node_features = G.ndata['h']
    edge_features = G.edata['h']
    labels = G.edata['Label']
    train_mask = G.edata['train_mask']
    test_mask = G.edata['test_mask']

    model = EGraphSAGE(
        ndim_in=edge_features.shape[2],
        ndim_out=128,
        edim=edge_features.shape[2],
        activation=F.relu,
        dropout=0.2
    ).to(device)

    optimizer = th.optim.Adam(model.parameters())
    criterion = th.nn.CrossEntropyLoss()

    start_time = timeit.default_timer()

    model.train()
    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats(device)
    for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training ({device})"):
        pred = model(G, node_features, edge_features).to(device)
        loss = criterion(pred[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            acc = compute_accuracy(pred[train_mask], labels[train_mask])
            tqdm.write(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

    train_time = timeit.default_timer() - start_time
    print(f"\n⏱️ Tempo di addestramento su {device}: {train_time:.2f} secondi")

    start_time = timeit.default_timer()

    model.eval()
    with th.no_grad():
        preds = model(G, node_features, edge_features)
        preds_test = preds[test_mask].argmax(1).cpu()
        labels_test = labels[test_mask].cpu()

        report_dict = classification_report(labels_test, preds_test, target_names=["Benign", "Malicious"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        print("\n📊 Test set performance:")
        print(report_df)
    test_time = timeit.default_timer() - start_time
    if th.cuda.is_available():
        th.cuda.synchronize()  # Assicura che tutto sia completato
        peak_mem = th.cuda.max_memory_allocated(device) / 1024**2
        print(f"[GPU 🚀 Picco memoria GPU allocata: {peak_mem:.2f} MB")
    return model, train_time, test_time, peak_mem, report_df


def main():
    seed = int(os.environ.get("SEED", 42))
    protocols = os.environ.get("PROTOCOLS", "all")  # Cambia qui per testare altri protocolli
    if protocols == 'tcp':

        datasets = {
            'full': pd.read_csv('data/NUSW_NB15/UNSW-NB15_processed_tcp.csv'),
            'dos': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_dos.csv'),
            'fuzzers': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_fuzzers.csv'),
            'exploits': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_exploits.csv'),
            'generic': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_generic.csv'),
            'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_reconnaissance.csv'),
        }

    elif protocols == 'all':
        datasets = {                
                'full': pd.read_csv('data/NUSW_NB15/UNSW-NB15_processed.csv'),
                'dos': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_dos.csv'),
                'fuzzers': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_fuzzers.csv'),
                'exploits': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_exploits.csv'),
                'generic': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_generic.csv'),
                'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_reconnaissance.csv'),
                'analysis': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_analysis.csv'),
                'shellcode': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_shellcode.csv'),
                'backdoor': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_backdoor.csv'),
            }
    else:
        datasets = {
            'dos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_dos.csv'),
            'ddos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ddos.csv'),
            'backdoor': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_backdoor.csv'),
            'injection': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_injection.csv'),
            'password': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_password.csv'),
            'ransomware': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ransomware.csv'),
            'scanning': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_scanning.csv'),
            'xss': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_xss.csv'),
        }

    os.makedirs("GNN/models", exist_ok=True)
    os.makedirs("GNN/results", exist_ok=True)
    results = []
    print(f"\n🌱 Esecuzione con seed {seed}")
    for name, df in datasets.items():
        print(f"\n📦 Costruzione grafo per: {name.upper()}")
        start_time = timeit.default_timer()
        if protocols == 'ToN':
            G_dgl, _ = preprocess_ToN_dataset(df, scaler_type='standard')
        else:
            # G_dgl, _ = preprocess_NUSW_dataset(df, protocols, scaler_type='standard')
            G_dgl, _ = preprocess_NUSW_dataset_optimized(df, protocols, scaler_type='standard')
        graph_time = timeit.default_timer() - start_time
        print(f"⏱️ Tempo di preprocessamento ottimizzato: {graph_time:.2f} secondi")

        for device in ['cuda']:
            if device == 'cuda' and not th.cuda.is_available():
                print(f"⚠️ GPU non disponibile, skip '{name.upper()}' su CUDA.")
                continue

            print(f"\n🚀 Addestramento del modello '{name.upper()}' su {device.upper()}")
            model, train_time, test_time, peak_mem, report_df = train_egraphsage(G_dgl, num_epochs=200, lr=1e-3, device=device, verbose=True, seed=seed)

            accuracy = report_df.loc["accuracy", "f1-score"]
            f1_malicious = report_df.loc["Malicious", "f1-score"]
            precision = report_df.loc["Malicious", "precision"]
            recall = report_df.loc["Malicious", "recall"]

            results.append({
                "attack_cat": name,
                "device": 'cuda:2',
                "graph_time": round(graph_time, 2),
                "train_time": round(train_time, 2),
                "test_time": round(test_time, 4),
                "peak_memory": round(peak_mem, 2),
                "accuracy": round(accuracy, 5),
                "f1_malicious": round(f1_malicious, 5),
                "precision_malicious": round(precision, 5),
                "recall_malicious": round(recall, 5)
            })

            # th.save(model.state_dict(), f"GNN/models/model_{name}_{device}_{protocols}_{seed}.pth")
            # print(f"✅ Modello salvato in: GNN/models/model_{name}_{device}_{seed}.pth")
            print(f"⏱️ Tempo su {device.upper()}: {train_time:.2f}s")
            print(f"⏱️ Test Time su {device.upper()}: {test_time:.2f}s")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"GNN/results/training_results_{protocols}_{seed}.csv", index=False)
    print(f"\n📄 Risultati salvati in GNN/results/training_results_{protocols}_{seed}.csv")

if __name__ == "__main__":
    main()