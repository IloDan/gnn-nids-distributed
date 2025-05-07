from cuml.metrics import accuracy_score, confusion_matrix
from cuml.ensemble import RandomForestClassifier
import cupy as cp
import os
import json
from src.config import INPUT_SIZE
from src.dataload import X_train_cp, y_train_cp, X_test_cp, y_test_np, Timer
import torch
import torch.multiprocessing as mp
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Function to train a RandomForest model and save the best one
def train_model(rank, X, y, X_test, y_test, params, counter, device):
    # Initialize RandomForestClassifier with given parameters
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features=params['max_features'],
        n_streams=1,
        random_state=rank
    )

    # Train the Random Forest model
    model.fit(X, y)

    # Validate the model
    val_preds = model.predict(X_test)
    val_accuracy = accuracy_score(y_test, val_preds)
    print(f"Rank {rank}, Validation Accuracy: {val_accuracy}")

    # Save the model if it has good validation accuracy
    model_path = f"models/best_rf_model_{counter + rank}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save hyperparameters for reference
    hyperparams_path = f"models/model_{counter + rank}_hyperparams.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(params, f)

# Function to evaluate multiple models (ensemble) with majority voting
def evaluate_ensemble(X, y, num_models):
    # Load trained models for evaluation
    models = []
    for rank in range(num_models):
        model_path = f"models/best_rf_model_{rank}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded model {rank}")
        models.append(model)

    # Collect predictions from all models
    all_preds = cp.zeros((num_models, X.shape[0]), dtype=cp.int32)
    for idx, model in enumerate(models):
        all_preds[idx] = model.predict(X)

    # Majority voting using efficient approach
    print("Calculating ensemble predictions...")
    final_preds = cp.apply_along_axis(lambda x: cp.bincount(x).argmax(), axis=0, arr=all_preds)
    print("Ensemble predictions calculated")
    
    # Calculate accuracy
    ensemble_accuracy = accuracy_score(y, final_preds)
    print("Ensemble Accuracy: {:.2f}%".format(ensemble_accuracy * 100))

    cm = confusion_matrix(y, final_preds).get()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix Random_Committee')
    plt.savefig('confusion_matrix_RC.png')
    return ensemble_accuracy


# Parameters
input_size = INPUT_SIZE
num_models = 10  # Run more models for improved ensemble performance
num_gpus = torch.cuda.device_count()
max_models_per_gpu = 5  # Set the maximum number of models to run on a single GPU

# Distributed training function to spawn processes
def main_worker(rank, num_gpus, max_models_per_gpu, counter, X_train_cp, y_train_cp, X_test_cp, y_test_np):
    # Set CUDA_VISIBLE_DEVICES environment variable dynamically
    gpu_id = (rank // max_models_per_gpu) % num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Rank {rank} assigned to GPU {gpu_id}")

    param = {
        'n_estimators': np.random.randint(100, 500),  # Number of trees in the forest
        'max_depth': np.random.randint(10, 20),  # Maximum depth of the tree
        'max_features': np.random.uniform(0.6, 0.9)  # Maximum features to consider for split
    }

    train_model(rank, X_train_cp, y_train_cp, X_test_cp, y_test_np, param, counter, gpu_id)

# Spawn processes for independent model training in batches
if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Train models in batches if the total number exceeds GPU capacity
    models_trained = 0
    with Timer() as t:
        while models_trained < num_models:
            models_to_train = min(max_models_per_gpu * num_gpus, num_models - models_trained)
            print(f"Training models {models_trained} to {models_trained + models_to_train - 1}")
            mp.spawn(
                main_worker,
                args=(num_gpus, max_models_per_gpu, models_trained, X_train_cp, y_train_cp, X_test_cp, y_test_np),
                nprocs=models_to_train,
                join=True
            )
            models_trained += models_to_train

    # Evaluate the ensemble with majority voting
    ensemble_accuracy= evaluate_ensemble(X_test_cp, y_test_np, num_models)
    with open('accuracy_time.txt', 'a') as f:
        f.write(f"RC:{ensemble_accuracy*100:.4f}:{t.elapsed:.4f}\n")
