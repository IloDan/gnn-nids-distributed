import os
import time
import threading

import cupy
import cudf
import dask_cudf
import pandas as pd
import numpy as np

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from cuml.dask.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix
)

from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset

import pynvml


class GPUMemoryMonitor:
    """
    Campiona l'uso di memoria di un insieme di GPU e mantiene il massimo (peak).
    """
    def __init__(self, gpu_ids=(0, 1), interval=0.1):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.peak = {gpu: 0 for gpu in gpu_ids}
        self._stop_event = threading.Event()
        self._thread = None

    def _worker(self):
        while not self._stop_event.is_set():
            for gpu in self.gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
                mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                if mem_used > self.peak[gpu]:
                    self.peak[gpu] = mem_used
            time.sleep(self.interval)

    def start(self):
        pynvml.nvmlInit()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        pynvml.nvmlShutdown()

    def peak_mb(self):
        return {gpu: round(b / 1024 ** 2, 2) for gpu, b in self.peak.items()}


class Timer:
    def __enter__(self):
        self.tick = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.tock = time.time()
        self.elapsed = self.tock - self.tick


class NoAffinityLocalCUDACluster(LocalCUDACluster):
    """
    Disabilita il plugin CPUAffinity di dask-cuda (evita os.sched_setaffinity Errno 22 su Slurm/cgroups).
    """
    def new_worker_spec(self):
        spec = super().new_worker_spec()
        for ws in spec.values():
            plugins = ws["options"].get("plugins", set())
            ws["options"]["plugins"] = {p for p in plugins if p.__class__.__name__ != "CPUAffinity"}
        return spec


if __name__ == "__main__":
    datasets = {
        "full": pd.read_csv("data/NUSW_NB15/UNSW-NB15_splitted.csv"),
        "dos": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_dos.csv"),
        "fuzzers": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_fuzzers.csv"),
        "exploits": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_exploits.csv"),
        "generic": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_generic.csv"),
        "reconnaissance": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_reconnaissance.csv"),
        "analysis": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_analysis.csv"),
        "shellcode": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_shellcode.csv"),
        "backdoor": pd.read_csv("data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_backdoor.csv"),
    }

    # Se vuoi ToN, sostituisci sopra e cambia preprocess_* sotto.

    all_metrics = []

    # cluster una volta sola (meglio di ricrearlo ad ogni dataset)
    # CUDA_VISIBLE_DEVICES: opzionale, ma utile per essere sicuri che usa solo 0 e 1
    with NoAffinityLocalCUDACluster(
        n_workers=2,
        threads_per_worker=1,
        CUDA_VISIBLE_DEVICES="0,1",
        # RMM opzionale: se vuoi puoi attivarlo dopo
        # rmm_pool_size="20GB",
    ) as cluster, Client(cluster) as client:

        client.wait_for_workers(2)

        for name, df in datasets.items():
            X_train, y_train, X_test, y_test = preprocess_NUSW_dataset(df, scaler_type="standard")
            # X_train, y_train, X_test, y_test = preprocess_TON_dataset(df, scaler_type="standard")

            X_train = X_train.astype("float32")
            X_test = X_test.astype("float32")
            y_train = y_train.astype("int32")
            y_test = y_test.astype("int32")

            monitor = GPUMemoryMonitor(gpu_ids=(0, 1), interval=0.1)
            monitor.start()

            # pandas -> cudf -> dask_cudf
            X_train_dask = dask_cudf.from_cudf(cudf.from_pandas(X_train), npartitions=2).persist()
            y_train_dask = dask_cudf.from_cudf(cudf.from_pandas(y_train), npartitions=2).persist()
            X_test_dask = dask_cudf.from_cudf(cudf.from_pandas(X_test), npartitions=2).persist()

            rf = RandomForestClassifier(n_estimators=100, n_streams=1, random_state=0)

            with Timer() as train_timer:
                rf.fit(X_train_dask, y_train_dask)

            with Timer() as test_timer:
                y_pred = rf.predict(X_test_dask).compute().astype("int32")

            monitor.stop()
            peak_mem = monitor.peak_mb()

            # sklearn vuole numpy
            y_test_np = y_test.to_numpy()
            y_pred_np = y_pred.to_numpy()

            accuracy = accuracy_score(y_test_np, y_pred_np)
            bal_accuracy = balanced_accuracy_score(y_test_np, y_pred_np)
            f1 = f1_score(y_test_np, y_pred_np, pos_label=1)
            precision = precision_score(y_test_np, y_pred_np, pos_label=1)
            recall = recall_score(y_test_np, y_pred_np, pos_label=1)

            all_metrics.append({
                "attack_cat": name,
                "device": "cuda:2",
                "graph_time": 0,
                "train_time": round(train_timer.elapsed, 2),
                "test_time": round(test_timer.elapsed, 4),
                "peak_memory_gpu_0": peak_mem.get(0, 0.0),
                "peak_memory_gpu_1": peak_mem.get(1, 0.0),
                "accuracy": round(accuracy, 5),
                "bal_Accuracy": round(bal_accuracy, 5),
                "f1_malicious": round(f1, 5),
                "precision_malicious": round(precision, 5),
                "recall_malicious": round(recall, 5),
            })

            # cleanup GPU memory tra un dataset e l'altro
            try:
                rf._reset_forest_data()
            except Exception:
                pass

            del rf, X_train_dask, y_train_dask, X_test_dask, y_pred
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()

            # piccola pausa per far respirare il cluster
            time.sleep(1)

    pd.DataFrame(all_metrics).to_csv("rf_dask_metrics_ToN_peak.csv", index=False)
