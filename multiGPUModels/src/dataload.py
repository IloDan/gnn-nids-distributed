from src.config import DATA_PATH, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE, VAL_TEST_SPLIT

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from category_encoders import TargetEncoder




import time

class Timer:    
    def __enter__(self):
        self.tick = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.tock = time.time()
        self.elapsed = self.tock - self.tick

def preprocess_NUSW_dataset(df, scaler_type='standard'):
    selected_features = [
            'IPSrcType', 'IPDstType', 
            'SrcPortWellKnown', 'DstPortWellKnown', 
            'SrcPortRegistered', 'DstPortRegistered', 
            'SrcPortPrivate', 'DstPortPrivate',
            'dur', 'sbytes', 'dbytes', 'BytesPerPkt', 'PktPerSec', 'RatioOutIn',
            'Dload', 'Dintpkt', 'res_bdy_len',
            'synack', 'ackdat',
            'sttl', 'dttl', 'tcprtt',
            'ct_state_ttl', 'ct_srv_dst', 'ct_srv_src', 'ct_dst_src_ltm', 'dmeansz',
            'state', 'service',
            'proto',
            'Label',
            'train_mask'
        ]

    df = df[selected_features].copy()

    df = pd.get_dummies(df, columns=['state', 'service'], drop_first=True)

    # Split train_mask == 1
    df_train, df_test = df[df['train_mask'] == 1], df[df['train_mask'] == 0]
    df_train = df_train.drop(columns=['train_mask'])
    df_test = df_test.drop(columns=['train_mask'])
    # Scaling
    feature_cols = [col for col in df_train.columns if col not in [
        'srcip', 'dstip', 'sport', 'dsport', 'src_ip_port', 'dst_ip_port', 'proto', 'Label']]

    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    encoder = TargetEncoder(smoothing=0.6)
    df_train['proto'] = encoder.fit_transform(df_train['proto'], df_train['Label'])
    df_test['proto'] = encoder.transform(df_test['proto'])
    feature_cols.append('proto')

    # Split features and labels
    X_train = df_train[feature_cols]
    y_train = df_train[LABEL_COLUMN]
    X_test = df_test[feature_cols]
    y_test = df_test[LABEL_COLUMN]

    # Convert data to numpy arrays for PyTorch


    return X_train, y_train, X_test, y_test


def preprocess_TON_dataset(df, scaler_type='standard'):
    df = df.copy() 
    if 'attack_cat' in df.columns:
        df = df.drop(columns=['attack_cat'])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

    feature_cols = list(df_train.columns)
    feature_cols.remove('src_ip_port')
    feature_cols.remove('dst_ip_port')
    feature_cols.remove('Label')
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

#shuffle the data
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)    

    # Split features and labels
    X_train = df_train[feature_cols]
    y_train = df_train["Label"]
    X_test = df_test[feature_cols]
    y_test = df_test["Label"]
    #shuffle the data
    

    if y_train.nunique() < 2:
        raise ValueError("Il train-set contiene una sola classe; "
                         "unisci record benigni o scegli un dataset diverso.")

    # Convert data to numpy arrays for PyTorch


    return X_train, y_train, X_test, y_test

def load_data_dask_NUWS(df, scaler_type='standard'):
    # Preprocess the dataset
    X_train, y_train, X_test, y_test = preprocess_NUSW_dataset(df, scaler_type)
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.int32)
    y_test_np = y_test.to_numpy(dtype=np.int32)

    # ########cuPy#########
    # import cupy as cp
    # X_train_cp = cp.array(X_train_np)
    # X_test_cp = cp.array(X_test_np)
    # y_train_cp = cp.array(y_train)
    #######################

    # Creazione array Dask
    import dask.array as da
    X_train_dask = da.from_array(X_train_np, chunks=(len(X_train_np) // 2, X_train_np.shape[1]))
    y_train_dask = da.from_array(y_train_np, chunks=(len(y_train_np) // 2,))
    X_test_dask = da.from_array(X_test_np, chunks=(len(X_test_np) // 2, X_test_np.shape[1]))
    y_test_dask = da.from_array(y_test_np, chunks=(len(y_test_np) // 2,))
    return X_train_dask, y_train_dask, X_test_dask, y_test_dask
    # Further split test set into validation and final test sets
    # X_val, X_test_final, y_val, y_test_final = train_test_split(X_test_np, y_test_np, test_size=VAL_TEST_SPLIT, random_state=RANDOM_STATE)

def load_data_dask_ToN(df, scaler_type='standard'):
    # Preprocess the dataset
    X_train, y_train, X_test, y_test = preprocess_TON_dataset(df, scaler_type)
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.int32)
    y_test_np = y_test.to_numpy(dtype=np.int32)

    # ########cuPy#########
    # import cupy as cp
    # X_train_cp = cp.array(X_train_np)
    # X_test_cp = cp.array(X_test_np)
    # y_train_cp = cp.array(y_train)
    #######################

    # Creazione array Dask
    import dask.array as da
    X_train_dask = da.from_array(X_train_np, chunks=(len(X_train_np) // 2, X_train_np.shape[1]))
    y_train_dask = da.from_array(y_train_np, chunks=(len(y_train_np) // 2,))
    X_test_dask = da.from_array(X_test_np, chunks=(len(X_test_np) // 2, X_test_np.shape[1]))
    y_test_dask = da.from_array(y_test_np, chunks=(len(y_test_np) // 2,))
    return X_train_dask, y_train_dask, X_test_dask, y_test_dask
    # Further split test set into validation and final test sets
    # X_val, X_test_final, y_val, y_test_final = train_test_split(X_test_np, y_test_np, test_size=VAL_TEST_SPLIT, random_state=RANDOM_STATE)

