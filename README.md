# Federated Learning for Healthcare Dataset

## Project Overview
This repository contains a complete PyTorch + Flower implementation for federated learning experiments on three healthcare-related datasets:
- UCI Heart Disease (tabular)
- Diabetes (Pima Indians, tabular)
- MNIST (baseline image classification)

The project supports:
- Centralized training baseline
- Federated learning with IID and non-IID splits
- Differential privacy on client-side training using Opacus
- Communication overhead measurement
- Convergence and privacy-utility analysis
- Optional blockchain logging of update metadata with Web3.py and Ganache/Hardhat

## Repository Structure
- `data/` - dataset download and partitioning scripts
- `src/` - core data, model, federated, DP, and utility modules
- `notebooks/` - experimental notebook for reproducibility
- `results/` - generated plots and CSV summaries
- `blockchain/` - smart contract and logging scripts
- `run_all_experiments.py` - main experiment runner
- `report.md` - submission-ready research report
- `demo_video_script.md` - demo narrative for presentation

## Setup Instructions
1. Create a Python environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download datasets and create partitions:
   ```bash
   python data/download_and_prepare.py
   python data/create_partitions.py
   ```

3. Run the full experiment suite:
   ```bash
   python run_all_experiments.py
   ```

4. View results in `results/` and read the research report in `report.md`.

## Experiment Variants
The runner executes:
- Centralized training baseline for each dataset
- `FL-IID` federated learning
- `FL-nonIID` federated learning using Dirichlet alpha=0.3
- `FL-label-skew` federated learning
- `FL+DP` on client-side training

## Blockchain Logging (Optional)
The blockchain module contains a smart contract and scripts to log model update hashes:
- `blockchain/contract/FLUpdateLogger.sol`
- `blockchain/scripts/deploy_logger.py`
- `blockchain/scripts/log_update.py`

A local Ethereum development chain such as Ganache or Hardhat can record metadata for model updates.

## Notes
- Random seeds are fixed for reproducibility.
- All plots are saved as high-resolution PNGs in `results/`.
- The code uses PyTorch models and Flower simulation.
