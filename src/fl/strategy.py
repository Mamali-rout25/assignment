from typing import Callable, Dict, List, Optional, Tuple
import flwr as fl
import numpy as np


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 4,
        min_evaluate_clients: int = 4,
        min_available_clients: int = 4,
        evaluate_fn=None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
        )
        self.communication: Dict[str, float] = {
            "downloaded_bytes": 0,
            "uploaded_bytes": 0,
            "rounds": 0,
        }

    @staticmethod
    def _parameters_size(parameters: fl.common.Parameters) -> int:
        arrays = fl.common.parameters_to_ndarrays(parameters)
        return sum(arr.nbytes for arr in arrays)

    def configure_fit(self, rnd, parameters, client_manager):
        config = super().configure_fit(rnd, parameters, client_manager)
        if config is None:
            return None
        selected = config[0]
        size = self._parameters_size(parameters)
        self.communication["downloaded_bytes"] += size * len(selected)
        self.communication["rounds"] += 1
        return config

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ) -> Optional[fl.common.Parameters]:
        if failures:
            return None
        total_upload = 0
        for res in results:
            total_upload += self._parameters_size(res.parameters)
        self.communication["uploaded_bytes"] += total_upload
        return super().aggregate_fit(rnd, results, failures)
