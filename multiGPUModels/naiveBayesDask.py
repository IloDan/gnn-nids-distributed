from cuml.dask.naive_bayes import MultinomialNB
import asyncio
from dask.distributed import Client, LocalCluster, Nanny
import cupy
import os
import dask_cudf
import cudf

from sklearn.metrics import accuracy_score, confusion_matrix
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
        worker = await Nanny(
            scheduler_address,
            env={"CUDA_VISIBLE_DEVICES": str(i)},
            resources=resource
        )
        workers.append(worker)
    return workers



if __name__ == "__main__":

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

    X_train_dask = dask_cudf.from_cudf(X_train, npartitions=2)
    X_test_dask = dask_cudf.from_cudf(X_test, npartitions=2)
    y_train_dask = dask_cudf.from_cudf(y_train, npartitions=2)
    y_test_dask = dask_cudf.from_cudf(y_test, npartitions=2)
    # Convert Dask-cuDF to Dask arrays
    X_train_dask_array = X_train_dask.to_dask_array(lengths=True)
    X_test_dask_array = X_test_dask.to_dask_array(lengths=True)
    y_train_dask_array = y_train_dask.to_dask_array(lengths=True)
    y_test_dask_array = y_test_dask.to_dask_array(lengths=True)
    model = MultinomialNB()
    with Timer() as t:
        # Train the model
        model.fit(X_train_dask_array, y_train_dask_array)
    
    # Compute accuracy on training set
    train_acc = model.score(X_train_dask_array, y_train_dask_array)
    y_pred_GNB_gpu = model.predict(X_test_dask_array)
    # Compute accuracy on test set
    test_acc = model.score(X_test_dask_array, y_test_dask_array)
    print(f"Training time: {t.elapsed}")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    # Convertire i risultati in NumPy arrays prima di calcolare metriche
    y_test_np = y_test_dask_array.compute().get()  # Da Dask a CuPy, poi a NumPy
    y_pred_np = y_pred_GNB_gpu.compute().get()  # Da Dask a CuPy, poi a NumPy

    # Calcolare la metrica di accuratezza
    accuracy = accuracy_score(y_test_np, y_pred_np)
    print(f"Accuracy: {accuracy}")

    # Calcolare la confusion matrix
    cm = confusion_matrix(y_test_np, y_pred_np)

    # Visualizzare la matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_naive.png')
    plt.show()
    client.close()
    cluster.close()