import asyncio
from dask.distributed import Client, LocalCluster, Nanny
from cuml.dask.ensemble import RandomForestClassifier
import cudf
import dask_cudf
import os
import cupy

from cuml.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import time

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
    for i in range(2):  # create 2 workers, one per GPU
        resource = {f"gpu{i}": 1}
        if i == 0:
            worker = await Nanny(
                scheduler_address,
                env={"CUDA_VISIBLE_DEVICES": "0,1"},
                resources=resource
            )
        else:
            worker = await Nanny(
                scheduler_address,
                env={"CUDA_VISIBLE_DEVICES": "1,0"},
                resources=resource
            )
        workers.append(worker)
    return workers


if __name__ == '__main__':

    X_train = cudf.read_csv('X_train.csv')
    X_test = cudf.read_csv('X_test.csv')
    y_train = X_train['label']
    y_test = X_test['label']
    X_train.drop('label', axis=1, inplace=True)
    X_test.drop('label', axis=1, inplace=True)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    # Start cluster and client
    cluster = LocalCluster(n_workers=0, threads_per_worker=1)
    client = Client(cluster)

    # Start workers asynchronously
    asyncio.run(start_workers(cluster.scheduler.address))
    client.wait_for_workers(2)


    # Check GPU details on each worker
    gpu_details = client.run(get_gpu_details)
    print("Detailed GPU info per worker:", gpu_details)

    
    # # Convert data to Dask-cuDF for multi-GPU processing
    # X_train_dask = dask_cudf.from_cudf(cudf.from_pandas(X_train), npartitions=2)
    # X_test_dask = dask_cudf.from_cudf(cudf.from_pandas(X_test), npartitions=2)
    # y_train_dask = dask_cudf.from_cudf(cudf.from_pandas(y_train.astype(np.int32)), npartitions=2)
    # y_test_dask = dask_cudf.from_cudf(cudf.from_pandas(y_test.astype(np.int32)), npartitions=2)
    X_train_dask = dask_cudf.from_cudf(X_train, npartitions=2)
    X_test_dask = dask_cudf.from_cudf(X_test, npartitions=2)
    y_train_dask = dask_cudf.from_cudf(y_train, npartitions=2)
    y_test_dask = dask_cudf.from_cudf(y_test, npartitions=2)


    X_train_dask = X_train_dask.persist()
    y_train_dask = y_train_dask.persist()

    
    # Train a RandomForestClassifier using cuML with multi-GPU support
    rf = RandomForestClassifier(n_estimators=100, n_streams=1, random_state=0)
    with Timer() as t:
        rf.fit(X_train_dask, y_train_dask)


    X_test_dask = X_test_dask.persist()
    y_test_dask = y_test_dask.persist()

    y_pred = rf.predict(X_test_dask).compute()

    # Calcolo accuratezza
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza sul set di test: {accuracy:.7f}")
    print(f"Tempo impiegato per l'addestramento: {t.elapsed:.7f} secondi")
    #save accuracy and time appened to file with name of the model the : is used to separate the values
    with open('accuracy_time.txt', 'a') as f:
        f.write(f"RF:{accuracy*100:.4f}:{t.elapsed:.4f}\n")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred).get()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix RF')
    plt.savefig('confusion_matrix_RF.png')

    # Explicitly release model resources to avoid issues during context shutdown
    rf._reset_forest_data()  # Attempting to explicitly release resources if available
    del rf

    client.shutdown()
    client.close()
    cluster.close()





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

