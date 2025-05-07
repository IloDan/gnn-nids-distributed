from src.config import DATA_PATH, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE, VAL_TEST_SPLIT

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.preprocessing import TargetEncoder



import time

class Timer:    
    def __enter__(self):
        self.tick = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.tock = time.time()
        self.elapsed = self.tock - self.tick

# Load dataset
df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns
df.drop(['attack_cat', 'id'], axis=1, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# One-hot encoding of categorical columns
df = pd.get_dummies(df, columns=['service', 'state'], drop_first=True)

# Split dataset into features and labels
X = df.drop(LABEL_COLUMN, axis=1)
y = df[LABEL_COLUMN]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Standardize continuous features
CONTINUOUS_FEATURES = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()

X_train[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES]).astype(np.float32)
X_test[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES]).astype(np.float32)

# Target encoding for categorical features
encoder = TargetEncoder(smooth=0.6)
X_train['proto'] = encoder.fit_transform(X_train['proto'], y_train).astype(np.float32)
X_test['proto'] = encoder.transform(X_test['proto']).astype(np.float32)

# Convert data to numpy arrays for PyTorch
X_train_np = X_train.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)
y_train_np = y_train.to_numpy(dtype=np.int32)
y_test_np = y_test.to_numpy(dtype=np.int32)
########cuPy#########
import cupy as cp
X_train_cp = cp.array(X_train_np)
X_test_cp = cp.array(X_test_np)
y_train_cp = cp.array(y_train)
#######################

# Creazione array Dask
import dask.array as da
X_train_dask = da.from_array(X_train_np, chunks=(len(X_train_np) // 2, X_train_np.shape[1]))
y_train_dask = da.from_array(y_train_np, chunks=(len(y_train_np) // 2,))
X_test_dask = da.from_array(X_test_np, chunks=(len(X_test_np) // 2, X_test_np.shape[1]))
y_test_dask = da.from_array(y_test_np, chunks=(len(y_test_np) // 2,))

# Further split test set into validation and final test sets
X_val, X_test_final, y_val, y_test_final = train_test_split(X_test_np, y_test_np, test_size=VAL_TEST_SPLIT, random_state=RANDOM_STATE)


# Code ready for training and evaluation pipelines.
