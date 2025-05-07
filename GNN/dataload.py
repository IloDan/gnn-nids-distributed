from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch as th
import networkx as nx
from dgl import from_networkx, graph
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder


def preprocess_NUSW_dataset(df, protocols='tcp', scaler_type='standard'):
    """
    Preprocessing completo per la costruzione del grafo DGL da un dataframe di attacchi.
    Restituisce il grafo DGL con edge features scalate, label, train_mask, test_mask.
    """
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
        'srcip', 'dstip', 'sport', 'dsport',
        'proto',
        'Label'
    ]

    # 1. Seleziona solo le feature desiderate
    df = df[selected_features].copy()

    df['src_ip_port'] = df['srcip'].astype(str) + ':' + df['sport'].astype(str)
    df['dst_ip_port'] = df['dstip'].astype(str) + ':' + df['dsport'].astype(str)

    # 2. One-hot encoding delle categoriche
    df = pd.get_dummies(df, columns=['state', 'service'], drop_first=True)

    # 3. Split train/test stratificato su Label
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

    # 4. Scaling
    feature_cols = list(df_train.columns)
    feature_cols.remove('srcip')
    feature_cols.remove('dstip')
    feature_cols.remove('sport')
    feature_cols.remove('dsport')
    feature_cols.remove('src_ip_port')
    feature_cols.remove('dst_ip_port')
    feature_cols.remove('proto')
    feature_cols.remove('Label')
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    #se protocols diverso da all
    if protocols != 'all':
        #Filtra solo i protocolli desiderati
        print(f"Filtering protocols: {protocols}")
        df_train = df_train[df_train['proto'] == protocols]
        df_test = df_test[df_test['proto'] == protocols]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        # 4.2 Rimuovi la colonna proto
        df_train = df_train.drop(columns=['proto'])
        df_test = df_test.drop(columns=['proto'])
    else:
        # Target Encoding Categorical Features
        encoder = TargetEncoder(smoothing=0.6)
        df_train['proto'] = encoder.fit_transform(df_train['proto'], df_train['Label'])
        df_test['proto'] = encoder.transform(df_test['proto'])
        feature_cols.append('proto')

    # 5. Ricostruisci i dataframe con colonne per il grafo
    df_train['h'] = df_train[feature_cols].values.tolist()

    df_test['h'] = df_test[feature_cols].values.tolist()
    df_train['train_mask'] = 1
    df_test['train_mask'] = 0

    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

    # 6. Crea grafo NetworkX e converti in DGL
    G_nx = nx.from_pandas_edgelist(df_all,
                                   source='src_ip_port',
                                   target='dst_ip_port',
                                   edge_attr=['h', 'Label', 'train_mask'],
                                   create_using=nx.MultiGraph())
    G_nx = G_nx.to_directed()
    print(f"Number of nodes in train: {len(list(G_nx.nodes))}")
    print(f"Number of edges in train: {len(list(G_nx.edges))}")

    G_dgl = from_networkx(G_nx, edge_attrs=['h', 'Label', 'train_mask'])

    G_dgl.ndata['h'] = th.ones(G_dgl.num_nodes(), G_dgl.edata['h'].shape[1])

    G_dgl.ndata['h'] = th.reshape(G_dgl.ndata['h'], (G_dgl.ndata['h'].shape[0], 1, G_dgl.ndata['h'].shape[1]))
    G_dgl.edata['h'] = th.reshape(G_dgl.edata['h'], (G_dgl.edata['h'].shape[0], 1, G_dgl.edata['h'].shape[1]))

    # 8. Crea test_mask inversa
    G_dgl.edata['train_mask'] = G_dgl.edata['train_mask'].bool()
    G_dgl.edata['test_mask'] = ~G_dgl.edata['train_mask']

    return G_dgl, scaler


def preprocess_ToN_dataset(df, scaler_type='standard'):
    """
    Preprocessing completo per la costruzione del grafo DGL da un dataframe di attacchi.
    Restituisce il grafo DGL con edge features scalate, label, train_mask, test_mask.
    """

    df = df.copy() 
    #rimuovi colonna attack_cat
    if 'attack_cat' in df.columns:
        df = df.drop(columns=['attack_cat'])
    # 3. Split train/test stratificato su Label
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

    # 4. Scaling
    feature_cols = list(df_train.columns)
    feature_cols.remove('src_ip_port')
    feature_cols.remove('dst_ip_port')
    feature_cols.remove('Label')
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # 5. Ricostruisci i dataframe con colonne per il grafo
    df_train['h'] = df_train[feature_cols].values.tolist()
    df_test['h'] = df_test[feature_cols].values.tolist()

    df_train['train_mask'] = 1
    df_test['train_mask'] = 0

    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

    # 6. Crea grafo NetworkX e converti in DGL
    G_nx = nx.from_pandas_edgelist(df_all,
                                   source='src_ip_port',
                                   target='dst_ip_port',
                                   edge_attr=['h', 'Label', 'train_mask'],
                                   create_using=nx.MultiGraph())
    G_nx = G_nx.to_directed()
    print(f"Number of nodes in train: {len(list(G_nx.nodes))}")
    print(f"Number of edges in train: {len(list(G_nx.edges))}")

    G_dgl = from_networkx(G_nx, edge_attrs=['h', 'Label', 'train_mask'])

    G_dgl.ndata['h'] = th.ones(G_dgl.num_nodes(), G_dgl.edata['h'].shape[1])

    G_dgl.ndata['h'] = th.reshape(G_dgl.ndata['h'], (G_dgl.ndata['h'].shape[0], 1, G_dgl.ndata['h'].shape[1]))
    G_dgl.edata['h'] = th.reshape(G_dgl.edata['h'], (G_dgl.edata['h'].shape[0], 1, G_dgl.edata['h'].shape[1]))

    # 8. Crea test_mask inversa
    G_dgl.edata['train_mask'] = G_dgl.edata['train_mask'].bool()
    G_dgl.edata['test_mask'] = ~G_dgl.edata['train_mask']

    return G_dgl, scaler


def preprocess_NUSW_dataset_optimized(df, protocols='tcp', scaler_type='standard'):
    """
    Preprocessing ottimizzato per costruire un grafo DGL dal dataset NUSW-NB15.
    Restituisce il grafo DGL con edge features scalate, label, train/test mask.
    """
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
        'srcip', 'dstip', 'sport', 'dsport',
        'proto',
        'Label'
    ]

    df = df[selected_features].copy()
    df['src_ip_port'] = df['srcip'].astype(str) + ':' + df['sport'].astype(str)
    df['dst_ip_port'] = df['dstip'].astype(str) + ':' + df['dsport'].astype(str)

    df = pd.get_dummies(df, columns=['state', 'service'], drop_first=True)

    # Split stratificato
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

    # Scaling
    feature_cols = [col for col in df.columns if col not in [
        'srcip', 'dstip', 'sport', 'dsport', 'src_ip_port', 'dst_ip_port', 'proto', 'Label']]

    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    if protocols != 'all':
        df_train = df_train[df_train['proto'] == protocols]
        df_test = df_test[df_test['proto'] == protocols]
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_train = df_train.drop(columns=['proto'])
        df_test = df_test.drop(columns=['proto'])
    else:
        encoder = TargetEncoder(smoothing=0.6)
        df_train['proto'] = encoder.fit_transform(df_train['proto'], df_train['Label'])
        df_test['proto'] = encoder.transform(df_test['proto'])
        feature_cols.append('proto')

    df_train['train_mask'] = 1
    df_test['train_mask'] = 0
    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

    # Codifica nodi come interi
    # Crea dizionario unico IP:PORT → ID intero
    all_nodes = pd.Series(pd.concat([df_all['src_ip_port'], df_all['dst_ip_port']]).unique())
    node_id_map = {val: i for i, val in enumerate(all_nodes)}

    # Mappa i nodi in base al dizionario completo
    src_codes = df_all['src_ip_port'].map(node_id_map).to_numpy()
    dst_codes = df_all['dst_ip_port'].map(node_id_map).to_numpy()
    num_nodes = len(node_id_map)
    edge_feats_np = np.array(df_all[feature_cols].values, dtype=np.float32, copy=True)
    
    ###########
    labels = df_all['Label'].values
    train_mask = df_all['train_mask'].values.astype(bool)
    # Duplica archi: originali + inversi
    src_full = np.concatenate([src_codes, dst_codes])
    dst_full = np.concatenate([dst_codes, src_codes])

    # Duplica attributi edge
    edge_feats_full = np.vstack([edge_feats_np, edge_feats_np])
    labels_full = np.concatenate([labels, labels])
    train_mask_full = np.concatenate([train_mask, train_mask])
    ##############

    # Crea grafo diretto
    g = graph((src_codes, dst_codes), num_nodes=num_nodes)

    # Assegna edge features

    # g.edata['h'] = th.tensor(edge_feats_np).unsqueeze(1)  # [E, 1, F]
    # g.edata['Label'] = th.tensor(df_all['Label'].values, dtype=th.int64)
    # g.edata['train_mask'] = th.tensor(df_all['train_mask'].values.astype(bool))
    # g.edata['test_mask'] = ~g.edata['train_mask']
    g = graph((src_full, dst_full), num_nodes=len(node_id_map))
    g.edata['h'] = th.tensor(edge_feats_full).unsqueeze(1)
    g.edata['Label'] = th.tensor(labels_full)
    g.edata['train_mask'] = th.tensor(train_mask_full)
    g.edata['test_mask'] = ~g.edata['train_mask']

    # Dummy node features: all ones (as nel tuo codice originale)
    g.ndata['h'] = th.ones((g.num_nodes(), 1, edge_feats_np.shape[1]), dtype=th.float32)

    print(f"[INFO] Number of nodes: {g.num_nodes()}")
    print(f"[INFO] Number of edges: {g.num_edges()}")

    return g, scaler



# def preprocess_partitioned_graph_from_dataframe(df, rank, world_size, protocols='all', scaler_type='standard'):

#     selected_features = [
#         'IPSrcType', 'IPDstType', 
#         'SrcPortWellKnown', 'DstPortWellKnown', 
#         'SrcPortRegistered', 'DstPortRegistered', 
#         'SrcPortPrivate', 'DstPortPrivate',
#         'dur', 'sbytes', 'dbytes', 'BytesPerPkt', 'PktPerSec', 'RatioOutIn',
#         'Dload', 'Dintpkt', 'res_bdy_len',
#         'synack', 'ackdat',
#         'sttl', 'dttl', 'tcprtt',
#         'ct_state_ttl', 'ct_srv_dst', 'ct_srv_src', 'ct_dst_src_ltm', 'dmeansz',
#         'state', 'service',
#         'srcip', 'dstip', 'sport', 'dsport',
#         'proto',
#         'Label'
#     ]

#     # 1. Seleziona solo le feature desiderate
#     df = df[selected_features].copy()
#     df['src_ip_port'] = df['srcip'].astype(str) + ':' + df['sport'].astype(str)
#     df['dst_ip_port'] = df['dstip'].astype(str) + ':' + df['dsport'].astype(str)

#     df = pd.get_dummies(df, columns=['state', 'service'], drop_first=True)

#     # Split train/test
#     df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
#     # 4. Scaling
#     feature_cols = list(df_train.columns)
#     feature_cols.remove('srcip')
#     feature_cols.remove('dstip')
#     feature_cols.remove('sport')
#     feature_cols.remove('dsport')
#     feature_cols.remove('src_ip_port')
#     feature_cols.remove('dst_ip_port')
#     feature_cols.remove('proto')
#     feature_cols.remove('Label')
#     scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
#     df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
#     df_test[feature_cols] = scaler.transform(df_test[feature_cols])
#     #se protocols diverso da all
#     if protocols != 'all':
#         #Filtra solo i protocolli desiderati
#         print(f"Filtering protocols: {protocols}")
#         df_train = df_train[df_train['proto'] == protocols]
#         df_test = df_test[df_test['proto'] == protocols]
#         df_train = df_train.reset_index(drop=True)
#         df_test = df_test.reset_index(drop=True)
#         # 4.2 Rimuovi la colonna proto
#         df_train = df_train.drop(columns=['proto'])
#         df_test = df_test.drop(columns=['proto'])
#     else:
#         # Target Encoding Categorical Features
#         encoder = TargetEncoder(smooth=0.6)
#         df_train['proto'] = encoder.fit_transform(df_train['proto'], df_train['Label'])
#         df_test['proto'] = encoder.transform(df_test['proto'])
#         feature_cols.append('proto')

#     # 5. Ricostruisci i dataframe con colonne per il grafo
#     df_train['h'] = df_train[feature_cols].values.tolist()

#     df_test['h'] = df_test[feature_cols].values.tolist()
#     df_train['train_mask'] = 1
#     df_test['train_mask'] = 0
#     df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

#     # 2. STRATIFIED PARTITIONING on the DataFrame directly
#     labels = df_all['Label'].values
#     benign_df = df_all[df_all['Label'] == 0]
#     malicious_df = df_all[df_all['Label'] == 1]

#     benign_split = len(benign_df) // world_size
#     malicious_split = len(malicious_df) // world_size

#     benign_part = benign_df.iloc[rank * benign_split : (rank + 1) * benign_split]
#     malicious_part = malicious_df.iloc[rank * malicious_split : (rank + 1) * malicious_split]
#     df_local = pd.concat([benign_part, malicious_part])


#     # 4. Crea il grafo locale
#     G_nx = nx.from_pandas_edgelist(
#         df_local,
#         source='src_ip_port',
#         target='dst_ip_port',
#         edge_attr=['h', 'Label', 'train_mask'],
#         create_using=nx.MultiGraph()
#     )
#     G_nx = G_nx.to_directed()
#     print(f"Number of nodes in train: {len(list(G_nx.nodes))}")
#     print(f"Number of edges in train: {len(list(G_nx.edges))}")

#     G_dgl = from_networkx(G_nx, edge_attrs=['h', 'Label', 'train_mask'])

#     G_dgl.ndata['h'] = th.ones(G_dgl.num_nodes(), G_dgl.edata['h'].shape[1])

#     G_dgl.ndata['h'] = th.reshape(G_dgl.ndata['h'], (G_dgl.ndata['h'].shape[0], 1, G_dgl.ndata['h'].shape[1]))
#     G_dgl.edata['h'] = th.reshape(G_dgl.edata['h'], (G_dgl.edata['h'].shape[0], 1, G_dgl.edata['h'].shape[1]))

#     # 8. Crea test_mask inversa
#     G_dgl.edata['train_mask'] = G_dgl.edata['train_mask'].bool()
#     G_dgl.edata['test_mask'] = ~G_dgl.edata['train_mask']

#     return G_dgl

