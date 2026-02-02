import ast
import asyncio
import os
import time
import cupy
import cudf
import dask_cudf
import pandas as pd
from src.utils import get_qfs_target_feature_k_rows
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dask.distributed import Client, LocalCluster, Nanny
from cuml.dask.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset



import threading
import pynvml 

class GPUMemoryMonitor:
    """
    Campiona l'uso di memoria di un insieme di GPU
    e mantiene il valore massimo (peak) osservato.
    """
    def __init__(self, gpu_ids=(0, 1), interval=0.1):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.peak = {gpu: 0 for gpu in gpu_ids}
        self._stop_event = threading.Event()

    def _worker(self):
        while not self._stop_event.is_set():
            for gpu in self.gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
                mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used  # byte
                if mem_used > self.peak[gpu]:
                    self.peak[gpu] = mem_used
            time.sleep(self.interval)

    def start(self):
        pynvml.nvmlInit()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        pynvml.nvmlShutdown()

    def peak_mb(self):
        # converte in MiB e arrotonda a due decimali
        return {gpu: round(b / 1024 ** 2, 2) for gpu, b in self.peak.items()}





class Timer:
    def __enter__(self):
        self.tick = time.time()
        return self
    def __exit__(self, *args, **kwargs):
        self.tock = time.time()
        self.elapsed = self.tock - self.tick


def get_gpu_details():
    device = cupy.cuda.Device(0)
    props = cupy.cuda.runtime.getDeviceProperties(device.id)
    return {
        "device": device.id,
        "GPU_name": props["name"],
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    }


async def start_workers(scheduler_address):
    workers = []
    for i in range(2):
        resource = {f"gpu{i}": 1}
        worker = await Nanny(
            scheduler_address,
            env={"CUDA_VISIBLE_DEVICES": str(i)},
            resources=resource
        )
        workers.append(worker)
    return workers


if __name__ == '__main__':
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
    
    all_qfs_features = pd.read_csv('./data/QFS_features.csv', index_col=0)
    selection_algorithms = ['QUBOCorrelation', 'QUBOMutualInformation', 'QUBOSVCBoosting']
    QUBO_solvers = ['SimulatedAnnealing', 'SteepestDescent', 'TabuSampler']

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results_file = "results/rf_dask_metrics_NUSW_peak.csv"
    try:
        results_df = pd.read_csv(results_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=[
            "attack_cat", "qfs_index", "selection_algorithm_name", "QUBO_solver",
            "target_feature_k", "device", "graph_time", "train_time", "test_time", "peak_memory_gpu_0", "peak_memory_gpu_1",
            "accuracy", "bal_accuracy", "f1_malicious", "precision_malicious", "recall_malicious"
        ])
    
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
                    
                    X_train, y_train, X_test, y_test = preprocess_NUSW_dataset(df, scaler_type='standard', qfs_features=selected_features)
                    # X_train, y_train, X_test, y_test = preprocess_TON_dataset(df, scaler_type='standard')
                    X_train = X_train.astype('float32')
                    X_test = X_test.astype('float32')
                    y_train = y_train.astype('int32')
                    y_test = y_test.astype('int32')

                    cluster = LocalCluster(n_workers=0, threads_per_worker=1)
                    client = Client(cluster)
                    asyncio.run(start_workers(cluster.scheduler.address))
                    client.wait_for_workers(2)
                    monitor = GPUMemoryMonitor(gpu_ids=[0, 1], interval=0.1)
                    monitor.start()
                    X_train_dask = dask_cudf.from_cudf(cudf.from_pandas(X_train), npartitions=2)
                    y_train_dask = dask_cudf.from_cudf(cudf.from_pandas(y_train), npartitions=2)
                    X_test_dask = dask_cudf.from_cudf(cudf.from_pandas(X_test), npartitions=2)  
                    nparts = 8                                  
                    # X_train_dask = (dask_cudf
                    #                 .from_cudf(cudf.from_pandas(X_train), npartitions=nparts)
                    #                 .shuffle(on='Label', npartitions=nparts))             
                    # y_train_dask = (dask_cudf
                    #                 .from_cudf(cudf.from_pandas(y_train), npartitions=nparts)
                    #                 .shuffle(on='Label', npartitions=nparts))              
                    # X_test_dask  =  dask_cudf.from_cudf(cudf.from_pandas(X_test),
                    #                                     npartitions=nparts)



                    X_train_dask = X_train_dask.persist()
                    y_train_dask = y_train_dask.persist()
                    
                    rf = RandomForestClassifier(n_estimators=100, n_streams=1, random_state=0)

                    with Timer() as train_timer:
                        rf.fit(X_train_dask, y_train_dask)


                    with Timer() as test_timer:
                        y_pred = rf.predict(X_test_dask).compute().astype('int32')
                    monitor.stop()
                    peak_mem = monitor.peak_mb()
                    y_test = y_test.to_numpy()
                    y_pred = y_pred.to_numpy()

                    accuracy = accuracy_score(y_test, y_pred)
                    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, pos_label=1)
                    precision = precision_score(y_test, y_pred, pos_label=1)
                    recall = recall_score(y_test, y_pred, pos_label=1)


                    cm = confusion_matrix(y_test, y_pred)
                    # plt.figure(figsize=(8, 6))
                    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
                    # plt.xlabel('Predicted labels')
                    # plt.ylabel('True labels')
                    # plt.title(f'Confusion Matrix RF - {name} - QFS {qfs_index:05d}')
                    # plt.savefig(f'figures/confusion_matrix_RF_{name}_QFS_{qfs_index:05d}_NUSW.png')

                    metrics_df = pd.DataFrame([{
                        "attack_cat": name,
                        "qfs_index": qfs_index,
                        "selection_algorithm_name": selection_algorithm,
                        "QUBO_solver": QUBO_solver,
                        "target_feature_k": feature_k,
                        "device": "cuda:2",
                        "graph_time": 0,
                        "train_time": round(train_timer.elapsed, 2),
                        "test_time": round(test_timer.elapsed, 4),
                        "peak_memory_gpu_0": peak_mem[0],
                        "peak_memory_gpu_1": peak_mem[1],
                        "accuracy": round(accuracy, 5),
                        "bal_accuracy": round(bal_accuracy, 5),
                        "f1_malicious": round(f1, 5),
                        "precision_malicious": round(precision, 5),
                        "recall_malicious": round(recall, 5)
                    }])

                    results_df = pd.concat([results_df, metrics_df], ignore_index=True)

                    rf._reset_forest_data()
                    cupy.get_default_memory_pool().free_all_blocks()
                    del rf
                    client.shutdown()
                    client.close()
                    time.sleep(2)
                    cluster.close()

                results_df.to_csv(results_file, index=False)
                # pd.DataFrame(all_metrics).to_csv("results/rf_dask_metrics_ToN_peak.csv", index=False)
                print(f"\nMetriche salvate in {results_file}")





# from src.dataload import X_train, X_test, y_train, y_test, Timer


# from cuml.dask.ensemble import RandomForestClassifier
# import cudf
# import dask_cudf
# import numpy as np
# import os
# import cupy
# from distributed import SpecCluster, Scheduler, Nanny
# from distributed import Client
# import logging

# if __name__ == "__main__":
    
#     # Convert data to Dask-cuDF for multi-GPU processing
#     X_train_dask = dask_cudf.from_cudf(cudf.from_pandas(X_train), npartitions=4)
#     X_test_dask = dask_cudf.from_cudf(cudf.from_pandas(X_test), npartitions=4)
#     y_train_dask = dask_cudf.from_cudf(cudf.from_pandas(y_train.astype(np.int32)), npartitions=4)
#     y_test_dask = dask_cudf.from_cudf(cudf.from_pandas(y_test.astype(np.int32)), npartitions=4)


#     client = Client(cluster)

#     # Now each worker sees only the GPU you specified.
#     # Proceed with your multi-GPU code here...
#     print("Workers:", cluster.workers)
    
#     # Confirm each worker sees only one GPU
#     # print("Cluster:", cluster)
#     # print("Client:", client)
#     # crea un file gpu_(id_gpu).txt con scritto l'id della gpu in uso
#     with open(f"gpu_{cupy.cuda.Device().id}.txt", "w") as f:
#         f.write(str(cupy.cuda.Device().id))

    
#     print(f"Current GPU in use: {cupy.cuda.Device().id}")
#     # Train a RandomForestClassifier using cuML with multi-GPU support
#     rf = RandomForestClassifier(n_estimators=100, n_streams=1, random_state=0)
#     with Timer() as t:
#         rf.fit(X_train_dask, y_train_dask)
    
#     y_pred = rf.predict(X_test_dask)
#     # Compute accuracy
#     train_acc = rf.score(X_train_dask, y_train_dask)
#     accuracy = rf.score(X_test_dask, y_test_dask)
    
#     print(f"Training time: {t.elapsed}")
#     print(f"Train Accuracy: {train_acc}")
#     print(f"Test Accuracy: {accuracy}")
    
#     # Explicitly release model resources to avoid issues during context shutdown
#     rf._reset_forest_data()  # Attempting to explicitly release resources if available
    
#     del rf

