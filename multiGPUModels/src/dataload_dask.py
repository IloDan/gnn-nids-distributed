from src.config import DATA_PATH, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.preprocessing import TargetEncoder
import cudf
import dask_cudf

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
df.drop(['attack_cat', 'id'], axis=1, inplace=True)
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
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES]).astype(np.float32)
X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES]).astype(np.float32)

# Target encoding for categorical features
encoder = TargetEncoder(smooth=0.6)
X_train_scaled['proto'] = encoder.fit_transform(X_train_scaled[['proto']], y_train).astype(np.float32)
X_test_scaled['proto'] = encoder.transform(X_test_scaled[['proto']]).astype(np.float32)

# Convert data to Dask-cuDF for multi-GPU processing
X_train_dask = dask_cudf.from_cudf(cudf.from_pandas(X_train_scaled), npartitions=4)
X_test_dask = dask_cudf.from_cudf(cudf.from_pandas(X_test_scaled), npartitions=4)
y_train_dask = dask_cudf.from_cudf(cudf.from_pandas(y_train.astype(np.int32)), npartitions=4)
y_test_dask = dask_cudf.from_cudf(cudf.from_pandas(y_test.astype(np.int32)), npartitions=4)

# Convert Dask-cuDF to Dask arrays
X_train_dask_array = X_train_dask.to_dask_array(lengths=True)
X_test_dask_array = X_test_dask.to_dask_array(lengths=True)
y_train_dask_array = y_train_dask.to_dask_array(lengths=True)
y_test_dask_array = y_test_dask.to_dask_array(lengths=True)
