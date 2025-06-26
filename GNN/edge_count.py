
from dataload import  preprocess_NUSW_dataset_optimized, preprocess_ToN_dataset_optimized
import pandas as pd

# Load datasets
datasetsNB = {
        'full': pd.read_csv('data/NUSW_NB15/UNSW-NB15_processed.csv'),
        'dos': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_dos.csv'),
        'fuzzers': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_fuzzers.csv'),
        'exploits': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_exploits.csv'),
        'generic': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_generic.csv'),
        'reconnaissance': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_reconnaissance.csv'),
        'analysis': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_analysis.csv'),
        'shellcode': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_shellcode.csv'),
        'backdoor': pd.read_csv('data/NUSW_NB15/attack_cat/UNSW-NB15_backdoor.csv'),
}


datasetsTON = {
    'full': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_full.csv'),
    'dos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_dos.csv'),
    'ddos': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ddos.csv'),
    'backdoor': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_backdoor.csv'),
    'injection': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_injection.csv'),
    'password': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_password.csv'),
    'ransomware': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_ransomware.csv'),
    'scanning': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_scanning.csv'),
    'xss': pd.read_csv('data/ToN_IoT/attack_cat/ToN_IoT_xss.csv'),
}

# create a csv with the number of edges and nodes for each dataset
with open('data/edge_count.csv', 'w') as f:
    f.write('dataset,edges,nodes\n')
    for dataset in datasetsNB:
        g, scaler = preprocess_NUSW_dataset_optimized(datasetsNB[dataset])
        f.write(f'NUSW_{dataset},{g.num_edges()},{g.num_nodes()}\n')

    for dataset in datasetsTON:
        g, scaler = preprocess_ToN_dataset_optimized(datasetsTON[dataset])
        f.write(f'ToN_{dataset},{g.num_edges()},{g.num_nodes()}\n')
# print the number of edges and nodes for each dataset
with open('data/edge_count.csv', 'r') as f:
    for line in f:
        print(line.strip())
