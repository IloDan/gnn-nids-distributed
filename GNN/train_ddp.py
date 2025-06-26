import os
import time
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import dgl
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from dataload import  preprocess_NUSW_dataset_optimized, preprocess_ToN_dataset_optimized, preprocess_partitioned_dataset_optimized, preprocess_ToN_partitioned_dataset_optimized
from EGraphSAGE import EGraphSAGE, compute_accuracy
import seaborn as sns
import matplotlib.pyplot as plt


def set_seed(seed):
    import random
    import numpy as np
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def stratified_partition_graph_edges(G, rank, world_size):
    labels = G.edata['Label']
    benign_edges = (labels == 0).nonzero(as_tuple=True)[0]
    malicious_edges = (labels == 1).nonzero(as_tuple=True)[0]
    benign_edges = benign_edges[th.randperm(len(benign_edges))]
    malicious_edges = malicious_edges[th.randperm(len(malicious_edges))]
    benign_per_part = len(benign_edges) // world_size
    malicious_per_part = len(malicious_edges) // world_size
    benign_part = benign_edges[rank * benign_per_part : (rank + 1) * benign_per_part]
    malicious_part = malicious_edges[rank * malicious_per_part : (rank + 1) * malicious_per_part]
    selected_edges = th.cat([benign_part, malicious_part])
    subgraph = dgl.edge_subgraph(G, selected_edges)
    return subgraph

def partition_graph_edges(G, rank, world_size):
    num_edges = G.num_edges()
    edge_ids = th.randperm(num_edges)  # shuffle all edges
    edges_per_part = num_edges // world_size
    start = rank * edges_per_part
    end = (rank + 1) * edges_per_part if rank != world_size - 1 else num_edges
    selected_edges = edge_ids[start:end]
    subgraph = dgl.edge_subgraph(G, selected_edges)
    return subgraph


def synchronize_node_embeddings(G, async_op=True):
    if dist.is_initialized() and dist.get_world_size() > 1:
        local_h = G.ndata['h']
        gathered_h = [th.zeros_like(local_h) for _ in range(dist.get_world_size())]
        if async_op:
            work = dist.all_gather(gathered_h, local_h, async_op=True)
            return work, gathered_h
        else:
            dist.all_gather(gathered_h, local_h)
            return None, gathered_h
    return None, None


def train(rank, world_size, G_full, protocols, num_epochs=100, lr=1e-3, save_path=None, seed=42):
    
    set_seed(seed)

    device = th.device(f'cuda:{rank}')
    if protocols == 'all' or protocols == 'ToN':
        G = partition_graph_edges(G_full, rank, world_size)
    else:
        G=G_full
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
        activation=nn.ReLU(),
        dropout=0.2
    ).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_train = time.perf_counter()

    model.train()
    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats(device)
    for epoch in range(num_epochs):
        pred = model(G, node_features, edge_features)
        loss = criterion(pred[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0 and epoch % 10 == 0:
            acc = compute_accuracy(pred[train_mask], labels[train_mask])
            print(f"[GPU {rank}] Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

    train_time = time.perf_counter() - start_train

    start_test = time.perf_counter()
    model.eval()
    with th.no_grad():
        preds = model(G, node_features, edge_features)
        preds_test = preds[test_mask].argmax(1).cpu()
        labels_test = labels[test_mask].cpu()
        report_dict = classification_report(labels_test, preds_test, target_names=["Benign", "Malicious"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        balanced_acc = balanced_accuracy_score(labels_test, preds_test)
        cm= confusion_matrix(labels_test, preds_test)

    # if save_path and rank == 0:
        # th.save(model.module.state_dict(), save_path)
    test_time = time.perf_counter() - start_test
    if th.cuda.is_available():
        th.cuda.synchronize()  # Assicura che tutto sia completato
        peak_mem = th.cuda.max_memory_allocated(device) / 1024**2
        print(f"[GPU {rank}] 🚀 Picco memoria GPU allocata: {peak_mem:.2f} MB")
    return train_time, test_time,  peak_mem, report_df, balanced_acc, cm


def main():
    seed = int(os.environ.get("SEED", 42))
    protocols = os.environ.get("PROTOCOLS", "part")
    if protocols == 'tcp':

        datasets = {
            'full': pd.read_csv('data/NUSW_NB15/UNSW-NB15_processed_tcp.csv'),
            'dos': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_dos.csv'),
            'fuzzers': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_fuzzers.csv'),
            'exploits': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_exploits.csv'),
            'generic': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_generic.csv'),
            'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat_tcp/UNSW-NB15_reconnaissance.csv'),

        }

    elif protocols == 'all' or protocols == 'part':
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
    os.makedirs("GNN/models", exist_ok=True)
    os.makedirs("GNN/results", exist_ok=True)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)
    
    for name, df in datasets.items():
        print(f"\n📦 Costruzione grafo per: {name.upper()}")
        start_time = time.perf_counter()
        if protocols == 'ToN':
            G_dgl, _ = preprocess_ToN_dataset_optimized(df, scaler_type='standard')
        elif protocols == 'part':
            G_dgl = preprocess_partitioned_dataset_optimized(df, rank, world_size, protocols)
        elif protocols == 'ToNpart':
            print("🚧 Preprocessing ToN partitioned dataset...")
            G_dgl = preprocess_ToN_partitioned_dataset_optimized(df, rank, world_size)
        else:
            G_dgl, _ = preprocess_NUSW_dataset_optimized(df, protocols, scaler_type='standard')

        graph_time = time.perf_counter() - start_time
        print(f"⏱️ Tempo di preprocessamento ottimizzato: {graph_time:.2f} secondi")
        print(f"\n🚀 Addestramento del modello '{name.upper()}'")
        save_path = f"GNN/models/model_{name}_{protocols}_ddp_{seed}.pth" 

        train_time, test_time,peak_mem, report_df, balanced_acc, cm = train(rank, world_size, G_dgl, protocols, 200, 1e-3, save_path, seed=seed)
        
        if rank == 0:
            accuracy = report_df.loc["accuracy", "f1-score"]
            f1_malicious = report_df.loc["Malicious", "f1-score"]
            precision = report_df.loc["Malicious", "precision"]
            recall = report_df.loc["Malicious", "recall"]
            print(f"✅Training Time: {train_time:.4f} seconds")
            print(f"✅Test Time: {test_time:.4f} seconds")
            results.append({
                "attack_cat": name,
                "device": 'cuda:2',
                "graph_time": round(graph_time, 2),
                "train_time": round(train_time, 2),
                "test_time": round(test_time, 4),
                "peak_memory": round(peak_mem, 2),
                "accuracy": round(accuracy, 5),
                "bal_Accuracy": round(balanced_acc, 5),
                "f1_malicious": round(f1_malicious, 5),
                "precision_malicious": round(precision, 5),
                "recall_malicious": round(recall, 5)
            })
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix Random_Committee')
            plt.savefig(f'ddp_confusion_matrix_gnn_{protocols}_{name}.png')
    cleanup()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"GNN/results/ddp_training_results_{protocols}_{seed}_cm .csv", index=False)
    print(f"\n📄 Risultati salvati in GNN/results/ddp_training_results_{protocols}_{seed}_cm.csv")

if __name__ == "__main__":
    main() #torchrun --nproc_per_node=2 --master_port=12355 GNN/train_ddp.py