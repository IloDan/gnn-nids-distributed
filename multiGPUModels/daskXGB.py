from src.dataload import X_train_dask, X_test_dask, y_train_dask, y_test_dask, y_test_np, Timer
from xgboost import dask as xgb
import asyncio
from dask.distributed import Client, LocalCluster, Nanny
import os
import cupy



# def get_gpu_details():
#     device = cupy.cuda.Device(0)
#     props = cupy.cuda.runtime.getDeviceProperties(device.id)
#     return {
#         "device": 'cuda',
#         "GPU_name": props["name"],
#         "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
#     }

# async def start_workers(scheduler_address):
#     workers = []
#     for i in range(2):  # create 2 workers, one per GPU
#         resource = {f"gpu{i}": 1}
#         worker = await Nanny(
#             scheduler_address,
#             env={"CUDA_VISIBLE_DEVICES": str(i), "XGBOOST_GPU_ID": str(i)},
#             resources=resource
#         )
#         workers.append(worker)
#     return workers

def train_xgb(client, dtrain):

    # Imposta i parametri, recuperando la GPU corretta dal worker
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'verbosity': 1
    }
    # Supponiamo che dtrain e dtest siano già definiti nel contesto del worker
    num_round = 100
    bst = xgb.train(
        client, 
        params, 
        dtrain, 
        num_boost_round=num_round,
        evals=[(dtrain, "train")]
    )
    return bst

if __name__ == "__main__":
    
    # Parametri del modello
    # params = {
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'tree_method': 'hist',
    #     'device': f'cuda:{int(os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))}',
    #     'verbosity': 1,
    # }
 

    # Start cluster and client
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)

    # # Start workers asynchronously
    # asyncio.run(start_workers(cluster.scheduler.address))
    # client.wait_for_workers(2)

    # Check GPU details on each worker
    # gpu_details = client.run(get_gpu_details)
    # print("Detailed GPU info per worker:", gpu_details)
    
        # Creazione DMatrix
    dtrain = xgb.DaskDMatrix(client, X_train_dask, y_train_dask)
    dtest = xgb.DaskDMatrix(client, X_test_dask, y_test_dask)
    with Timer() as t:
        bst=train_xgb(client,dtrain)


    
   
    # Predizioni
    y_pred_prob = xgb.predict(client, bst['booster'], dtest)
    y_pred_prob=y_pred_prob.compute()
    # Conversione delle probabilità in etichette binarie
    y_pred = (y_pred_prob > 0.5).astype(int)
    client.close()
    cluster.close()

    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Calcolo accuratezza
    accuracy = accuracy_score(y_test_np, y_pred)
    print(f"Accuratezza sul set di test: {accuracy:.7f}")
    print(f"Tempo impiegato per l'addestramento: {t.elapsed:.7f} secondi")

    # Confusion Matrix
    cm = confusion_matrix(y_test_np, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix_XGB')
    plt.savefig('confusion_matrix_XGB.png')
    with open('accuracy_time.txt', 'a') as f:
        f.write(f"XGB:{accuracy*100:.4f}:{t.elapsed:.4f}\n")