import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import json
import random
from torch.utils.data import DataLoader, TensorDataset

from src.config import INPUT_SIZE
from multiGPUModels.src.dataload import X_train_np, y_train_np, X_val, y_val, X_test_final, y_test_final, BATCH_SIZE

# Create PyTorch datasets and loaders
train_data = TensorDataset(torch.tensor(X_train_np), torch.tensor(y_train_np, dtype=torch.long))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)

# Test loader for final evaluation (if needed)
test_data = TensorDataset(torch.tensor(X_test_final), torch.tensor(y_test_final, dtype=torch.long))
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


# Define a simple neural network with random hyperparameters
class NetworkAttackClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, activation_fn):
        super(NetworkAttackClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], 2)  # Output size is 2 for binary classification
        self.activation_fn = activation_fn
        self.batchnorm1 = nn.BatchNorm1d(hidden_sizes[0])
        self.batchnorm2 = nn.BatchNorm1d(hidden_sizes[1])
        self.batchnorm3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.activation_fn(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation_fn(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation_fn(self.batchnorm3(self.fc3(x)))
        x = self.fc4(x)
        return x

# Function to train a single model and save the best one
def train_model(rank, model, train_loader, val_loader, criterion, optimizer, epochs, device, counter):
    torch.manual_seed(rank)
    model = model.to(device)
    model.train()
    patience = 5
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch", disable=(rank != 0)) as pbar:
            pbar.set_description(f"Epoch [{epoch+1}/{epochs}] (Rank {rank})")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / len(train_loader))
        
        # Validation step
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if rank == 0:
            print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}")

        # Save the model if the validation loss is the best we've seen
        if val_loss < best_val_loss-0.0001:
            print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}], Validation loss improved from {best_val_loss} to {val_loss} after {epochs_no_improve} epochs")
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (Rank {rank}) due to no improvement in validation loss.")
            break
    # Save the best model state to file
    torch.save(best_model_state, f"models/best_model_{counter+rank}.pth")

# Function to evaluate multiple models (ensemble) with weighted voting
def evaluate_ensemble(dataloader, device):
     # Load trained models for evaluation
    models = []
    for rank in range(num_models):
        with open(f"models/model_{rank}_hyperparams.json", 'r') as f:
            hyperparams = json.load(f)
        hidden_sizes = hyperparams['hidden_sizes']
        dropout_rate = hyperparams['dropout_rate']
        activation_fn = getattr(nn, hyperparams['activation_fn'])()
        model = NetworkAttackClassifier(input_size, hidden_sizes, dropout_rate, activation_fn)
        model.load_state_dict(torch.load(f"models/best_model_{rank}.pth", weights_only=True))
        models.append(model)
    all_outputs = []
    model_weights = []
    for model in models:
        model.eval()
        model.to(device)
        outputs_list = []
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                outputs_list.append(outputs.cpu())
        accuracy = correct / total
        print(f"Model accuracy: {accuracy}")
        model_weights.append(accuracy)  # Use accuracy as the weight for each model
        all_outputs.append(torch.cat(outputs_list, dim=0))
    
    # Normalize weights
    model_weights = torch.tensor(model_weights)
    model_weights = model_weights / model_weights.sum()
    
    # Weighted average of the predictions from all models
    weighted_outputs = torch.zeros_like(all_outputs[0])
    for weight, output in zip(model_weights, all_outputs):
        weighted_outputs += weight * output
    
    predicted_labels = torch.argmax(weighted_outputs, dim=1)
    return predicted_labels

# Distributed training function to spawn processes
def main_worker(rank, num_gpus, max_models_per_gpu, input_size, val_loader, epochs, counter):
    # Randomly initialize hyperparameters for each model
    hidden_sizes = [
        np.random.randint(128, 512),  # Hidden layer 1 size
        np.random.randint(64, 256),   # Hidden layer 2 size
        np.random.randint(32, 128)    # Hidden layer 3 size
    ]
    dropout_rate = np.random.uniform(0.3, 0.5)  # Random dropout rate between 0.3 and 0.5
    learning_rate = np.random.uniform(1e-4, 1e-2)  # Random learning rate between 0.0001 and 0.01
    activation_fn = random.choice([nn.ReLU(), nn.LeakyReLU(), nn.ELU()])  # Random activation function

    # Limit the number of models assigned to each GPU
    gpu_id = (rank // max_models_per_gpu) % num_gpus
    device_id = gpu_id
    print(f"Rank {rank} assigned to GPU {device_id}")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(rank)
    model = NetworkAttackClassifier(input_size, hidden_sizes, dropout_rate, activation_fn)
    criterion = nn.CrossEntropyLoss()
    #optimizer beetween SGD and Adam
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_model(rank, model, train_loader, val_loader, criterion, optimizer, epochs, device, counter)
    
    # Save hyperparameters for reference
    with open(f"models/model_{rank}_hyperparams.json", 'w') as f:
        hyperparams = {
            'hidden_sizes': hidden_sizes,
            'dropout_rate': dropout_rate,
            'activation_fn': activation_fn.__class__.__name__,
            'learning_rate': learning_rate
        }
        json.dump(hyperparams, f)

# Parameters
input_size = INPUT_SIZE
num_models = 10
epochs = 50
num_gpus = torch.cuda.device_count()
max_models_per_gpu = 5 # Set the maximum number of models to run on a single GPU

# Spawn processes for independent model training in batches
if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Train models in batches if the total number exceeds GPU capacity
    models_trained = 0
    while models_trained < num_models:
        models_to_train = min(max_models_per_gpu * num_gpus, num_models - models_trained)
        print(f"Training models {models_trained} to {models_trained + models_to_train - 1}")
        mp.spawn(main_worker, args=(num_gpus, max_models_per_gpu, input_size, val_loader, epochs, models_trained), 
                 nprocs=models_to_train, join=True)
        models_trained += models_to_train

    # Evaluate the ensemble with weighted voting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predicted_labels = evaluate_ensemble(val_loader, device)

    # Calculate accuracy for the ensemble
    true_labels = torch.cat([labels for _, labels in val_loader])
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Ensemble Accuracy:", accuracy)
