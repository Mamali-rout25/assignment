import os
import pandas as pd
from src.utils.experiment import ExperimentRunner


def main():
    os.makedirs("results", exist_ok=True)
    runner = ExperimentRunner(
        data_root="data",
        results_root="results",
        num_clients=4,
        num_rounds=5,
        local_epochs=2,
        central_epochs=10,
        dp_target_epsilon=8.0,
        dp_target_delta=1e-5,
        seed=42,
    )

    summary = runner.run_all()
    summary.to_csv(os.path.join("results", "summary_metrics.csv"), index=False)
    print("All experiments complete. Summary saved to results/summary_metrics.csv")


if __name__ == "__main__":
    main()
