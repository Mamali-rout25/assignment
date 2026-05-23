# Federated Learning for Healthcare Dataset

## 1. Introduction & Motivation
Federated learning (FL) enables collaborative model training across distributed clients while preserving data locality. Healthcare applications, especially tabular clinical datasets such as heart disease and diabetes, demand privacy-preserving solutions because patient information cannot be shared centrally. This project compares centralized and federated learning strategies on healthcare and image datasets, evaluates non-IID data heterogeneity, and integrates differential privacy to quantify privacy-utility trade-offs.

## 2. Methodology
### 2.1 Federated Learning Architecture
We use the Flower framework with PyTorch models. Four simulated clients train local models on disjoint data partitions and exchange gradients with a central aggregator using FedAvg.

### 2.2 Datasets
- UCI Heart Disease (binary classification)
- Pima Indians Diabetes (binary classification)
- MNIST (baseline digit classification)

### 2.3 Data Partitioning
Two partition strategies were implemented:
- IID partitioning across four clients
- non-IID partitioning using Dirichlet distribution with alpha=0.3
- label skew partitioning, where each client receives a limited number of label classes

### 2.4 Differential Privacy
Client-side training uses Opacus to enforce differential privacy. The privacy engine tracks noise multiplier and clipping norms for each local update.

### 2.5 Blockchain Logging
A smart contract logs model update hashes and metadata on a local Ethereum test network. This is an optional audit trail and proof-of-contribution mechanism.

## 3. Experimental Setup
### 3.1 Hyperparameters
- Number of clients: 4
- FedAvg rounds: 5
- Local epochs per round: 2
- Centralized epochs: 10
- Batch size: 32
- Learning rate: 0.001
- DP epsilon target: 8.0, delta: 1e-5
- Dirichlet alpha: 0.3

### 3.2 Evaluation
The evaluation metrics include accuracy, loss, and communication overhead in bytes transferred during FL rounds.

## 4. Results & Discussion
### 4.1 Accuracy Comparison
The experiment runner generates an accuracy comparison table for:
- Centralized baseline
- FL-IID
- FL-nonIID
- FL-label-skew
- FL+DP

### 4.2 Communication Overhead
Communication overhead is measured as the total bytes uploaded and downloaded during federated training.

### 4.3 Convergence Plots
Loss and accuracy curves for each method highlight training stability and convergence patterns.

### 4.4 Privacy-Utility Trade-off
Differential privacy introduces noise into local updates. The trade-off is captured by a small utility drop compared to the non-private FL variant, while providing formal privacy guarantees.

## 5. Conclusion & Future Work
FL is a strong alternative to centralized training for privacy-sensitive healthcare data. Non-IID distributions reduce convergence speed, but FL still achieves competitive performance. Differential privacy is effective with moderate epsilon values, and blockchain logging adds transparency to model update provenance.

### Future extensions
- run experiments with more clients and more rounds
- integrate secure aggregation
- deploy smart contract logging to IPFS for model CID storage
- evaluate on additional real-world healthcare datasets
