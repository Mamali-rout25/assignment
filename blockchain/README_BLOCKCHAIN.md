# Blockchain Logging Module

This folder contains a simple Ethereum smart contract and Python scripts to log federated learning update hashes on a local blockchain.

## Files
- `contract/FLUpdateLogger.sol`: Solidity contract to store update metadata.
- `scripts/deploy_logger.py`: compile and deploy the contract to a local node.
- `scripts/log_update.py`: log a model update hash and metadata after training.

## Setup
1. Start a local Ethereum node, e.g. Ganache or Hardhat:
   ```bash
   ganache --deterministic
   ```
2. Install dependencies if not already installed:
   ```bash
   pip install -r requirements.txt
   ```

## Deployment
```bash
python blockchain/scripts/deploy_logger.py
```

## Logging a Model Update
Set environment variables and execute:
```bash
set WEB3_PROVIDER=http://127.0.0.1:8545
set CONTRACT_INFO=blockchain/contract_info.json
set MODEL_PATH=results/model_update.pt
set DATASET=heart
set VARIANT=fl_noniid
set ROUND=1
python blockchain/scripts/log_update.py
```

## Notes
- The contract deployer script compiles using `solcx`.
- The contract address and ABI can be saved to `blockchain/contract_info.json` for later use.
