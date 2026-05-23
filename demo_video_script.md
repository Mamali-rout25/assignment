# Demo Video Script

## 1. Introduction (30 seconds)
- Introduce the project: "Federated Learning for Healthcare Dataset." 
- Mention datasets: Heart Disease, Diabetes, and MNIST baseline.
- Explain the goal: compare centralized training, FL-IID, FL-nonIID, and FL+DP.

## 2. Project Walkthrough (90 seconds)
- Show repository structure in VS Code.
- Open `README.md` and highlight setup instructions.
- Open `src/` to show modular architecture: data loaders, models, federated client/server, DP utilities.
- Open `data/create_partitions.py` and explain partition strategies.

## 3. Experiment Execution (60 seconds)
- Run `python run_all_experiments.py` or show the script contents.
- Note that the runner generates results in `results/`, including summary CSV and plots.
- Mention that repeatable experiments use fixed random seeds.

## 4. Results Showcase (60 seconds)
- Open `results/summary_metrics.csv` and describe the metrics columns.
- Show saved plots: accuracy/loss curves and communication overhead.
- Explain the privacy-utility trade-off for FL+DP.

## 5. Blockchain Logging (30 seconds)
- Open `blockchain/contract/FLUpdateLogger.sol` and explain the audit log contract.
- Describe `deploy_logger.py` and `log_update.py` for registering model update metadata on a local chain.

## 6. Conclusion (30 seconds)
- Summarize key findings: federated learning is effective on healthcare tabular tasks, non-IID data is more challenging, DP adds privacy protection with reasonable utility.
- Mention future work: secure aggregation, scaling to more clients, real-world hospital datasets.
