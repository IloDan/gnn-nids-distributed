# Serial Code Porting on HPC & Quantum Computing

This repository contains the code, scripts, and configurations used for the project described in my thesis:

**Danilo Caputo – “Distributed Training in Traditional ML-based and GNN-based NIDS – A comparative study of multi-GPU optimization for Graph Neural Networks and ensemble methods”**  
(UNIMORE, A.Y. 2024/2025)

The project focuses on porting a **serial** pipeline to **HPC** environments (multi-core / multi-GPU) and evaluating:
- **Traditional ML** models with distributed execution
- **GNN-based** models with distributed training
- an experimental **Quantum** track limited to **feature selection only** (no quantum training/inference)

---

## Project overview

### 1) HPC track (main contribution)
- Netflow **pre-processing**: cleaning, encoding, scaling, feature engineering, stratified splits
- **Traditional ML (tabular)**: GPU-based training with **cuML**, distributed execution with **Dask**
- **GNN**: multi-GPU distributed training with **PyTorch DistributedDataParallel (DDP)** (+ a GNN library)
- **Benchmarks**: training/testing time and scaling comparison (1 GPU vs 2 GPUs)

### 2) Quantum track (feature selection only)
- Experimental module for **feature selection**
- Output: ranked/subset list of features to be used by the classical/HPC pipeline
- **No** model training or inference is executed on quantum backends in this project

---

## Hardware & software (as in thesis)

Main experiments were executed on an HPC node with:
- CPU: Intel Xeon Gold 5215 (40 physical cores)
- RAM: 500 GiB
- GPU: 2× NVIDIA Tesla V100 (32 GiB each)
- OS: Ubuntu 20.04
- Python: 3.11
- CUDA toolkit: 12.3

Main stack:
- PyTorch + `torch.distributed` (DDP)
- Dask + cuML (distributed ML on GPU)
- a GNN library (e.g., DGL) for graph handling and training

---


## Datasets

The thesis experiments use netflow-based datasets such as:
- **UNSW-NB15**
- **ToN-IoT**

> Datasets are not included in this repository (license/size reasons).
Place them locally (e.g., `data/`) and update the paths in the config files.

---

## Scope 

- **HPC:** full porting + distributed training (traditional ML and GNN) + benchmarks
- **Quantum:** **feature selection only**  
  No training or inference is executed on quantum backends in this repository.

---

## Citation

If you use this repository in academic work, please cite:

Danilo Caputo, *Distributed Training in Traditional ML-based and GNN-based NIDS – A comparative study of multi-GPU optimization for Graph Neural Networks and ensemble methods*, UNIMORE, A.Y. 2024/2025.
