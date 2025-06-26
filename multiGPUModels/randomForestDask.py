import asyncio
import os
import time
import cupy
import cudf
import dask_cudf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dask.distributed import Client, LocalCluster, Nanny
from cuml.dask.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from src.dataload import preprocess_NUSW_dataset, preprocess_TON_dataset


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
        # 'full': pd.read_csv('data/NUSW_NB15/UNSW-NB15_splitted.csv'),
        # 'dos': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_dos.csv'),
        # 'fuzzers': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_fuzzers.csv'),
        # 'exploits': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_exploits.csv'),
        # 'generic': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_generic.csv'),
        # 'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_reconnaissance.csv'),
        # 'analysis': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_analysis.csv'),
        # 'shellcode': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_shellcode.csv'),
        # 'backdoor': pd.read_csv('data/NUSW_NB15/attack_cat_splitted/UNSW-NB15_backdoor.csv'),
    }

    datasets = {
        'full': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_full.csv'),
        'dos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_dos.csv'),
        # 'ddos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ddos.csv'),
        # 'backdoor': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_backdoor.csv'),
        # 'injection': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_injection.csv'),
        # 'password': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_password.csv'),
        # 'ransomware': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ransomware.csv'),
        # 'scanning': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_scanning.csv'),
        # 'xss': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_xss.csv'),
    }

    all_metrics = []

    for name, df in datasets.items():
        # X_train, y_train, X_test, y_test = preprocess_NUSW_dataset(df, scaler_type='standard')
        X_train, y_train, X_test, y_test = preprocess_TON_dataset(df, scaler_type='standard')
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = y_train.astype('int32')
        y_test = y_test.astype('int32')

        cluster = LocalCluster(n_workers=0, threads_per_worker=1)
        client = Client(cluster)
        asyncio.run(start_workers(cluster.scheduler.address))
        client.wait_for_workers(2)

        # X_train_dask = dask_cudf.from_cudf(cudf.from_pandas(X_train), npartitions=8)
        # y_train_dask = dask_cudf.from_cudf(cudf.from_pandas(y_train), npartitions=8)
        # X_test_dask = dask_cudf.from_cudf(cudf.from_pandas(X_test), npartitions=8)  
        nparts = 8                                   # ⇠ NEW  (≥ 4 × n_gpu)
        X_train_dask = (dask_cudf
                        .from_cudf(cudf.from_pandas(X_train), npartitions=nparts)
                        .shuffle(on='Label', npartitions=nparts))              # ⇠ NEW
        y_train_dask = (dask_cudf
                        .from_cudf(cudf.from_pandas(y_train), npartitions=nparts)
                        .shuffle(on='Label', npartitions=nparts))              # ⇠ NEW
        X_test_dask  =  dask_cudf.from_cudf(cudf.from_pandas(X_test),
                                            npartitions=nparts)



        X_train_dask = X_train_dask.persist()
        y_train_dask = y_train_dask.persist()
        
        rf = RandomForestClassifier(n_estimators=100, n_streams=1, random_state=0)

        with Timer() as train_timer:
            rf.fit(X_train_dask, y_train_dask)

        with Timer() as test_timer:
            y_pred = rf.predict(X_test_dask).compute().astype('int32')

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
        # plt.title(f'Confusion Matrix RF - {name}')
        # plt.savefig(f'confusion_matrix_RF_{name}_ToN.png')

        all_metrics.append({
            "attack_cat": name,
            "device": "cuda:2",
            "graph_time": 0,
            "train_time": round(train_timer.elapsed, 2),
            "test_time": round(test_timer.elapsed, 4),
            "accuracy": round(accuracy, 5),
            "bal_Accuracy": round(bal_accuracy, 5),
            "f1_malicious": round(f1, 5),
            "precision_malicious": round(precision, 5),
            "recall_malicious": round(recall, 5)
        })

        rf._reset_forest_data()
        del rf
        client.shutdown()
        client.close()
        time.sleep(2)
        cluster.close()

    pd.DataFrame(all_metrics).to_csv("rf_dask_metrics_ToN.csv", index=False)





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

