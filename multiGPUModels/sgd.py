from src.dataload import X_train_np, y_train_np, X_test_np, y_test_np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
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

# Conversione a tensor e GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataHandler:
    def __init__(self, X_train_np, y_train_np, X_test_np, y_test_np, device):
        self.device = device
        self.X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
        self.X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        self.y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

# Modello Logistic Regression
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single output for binary classification

    def forward(self, x):
        return self.linear(x)

# Wrapper per l'allenamento del modello
class ModelTrainer:
    def __init__(self, model, dataloader, criterion, optimizer, scheduler, num_epochs, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.t = None

    def train(self):
        pbar = tqdm(range(self.num_epochs), desc="Training Progress")
        with Timer() as self.t:
            for epoch in pbar:
                self.model.train()
                self.optimizer.zero_grad()
                X_train, y_train = self.dataloader.get_train_data()
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_postfix({"Loss": loss.item()})
        pbar.close()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            X_test, y_test = self.dataloader.get_test_data()
            test_outputs = self.model(X_test)
            predicted = (torch.sigmoid(test_outputs) > 0.5).float()  # Convert logits to class predictions (0 or 1)
            accuracy = (predicted == y_test).float().mean().item()
            print(f"Test Accuracy: {accuracy}")
             # Calcolo della matrice di confusione
        y_true = y_test.cpu().numpy().astype(int)
        y_pred = predicted.cpu().numpy().astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_sgd.png')
        with open('accuracy_time.txt', 'a') as f:
            f.write(f"SGD:{accuracy*100:.4f}:{self.t.elapsed:.4f}\n")

if __name__ == "__main__":
    # Data Handling
    data_handler = DataHandler(X_train_np, y_train_np, X_test_np, y_test_np, device)
    
    # Model Initialization
    model = LogisticRegressionModel(input_dim=data_handler.get_train_data()[0].shape[1]).to(device)
    
    # Wrap the model with DataParallel to use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    
    # Optimizer, Loss Function, Scheduler
    lr = 1.5
    weight_decay = 0.00001
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch * 0.00001))
    
    # Model Trainer
    num_epochs = 1500
    trainer = ModelTrainer(model, data_handler, criterion, optimizer, scheduler, num_epochs, device)
    trainer.train()
    trainer.evaluate()
