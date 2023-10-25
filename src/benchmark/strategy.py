from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from typing import List, Optional, Tuple, Union, Dict

import numpy as np

class SaveModelStrategy(FedAvg):
    #def aggregate_fit(
    #    self,
    #    server_round: int,
    #    results: List[Tuple[ClientProxy, FitRes]],
    #    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    #) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        #aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        #if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
        #    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
        #    print(f"Saving round {server_round} aggregated_ndarrays...")
        #    np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        #return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the received local parameters and store the test aggregated.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: Exceptions that occurred while the server
                was waiting for client updates.

        Returns
        -------
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round=server_round, results=results, failures=failures
        )
        _ = aggregated_metrics  # Avoid unused variable warning

        # Weigh accuracy of each client by number of examples used
        accuracies: List[float] = []
        for _, res in results:
            accuracy: float = float(res.metrics["accuracy"])
            accuracies.append(accuracy)
        print(f"Round {server_round} accuracies: {accuracies}")

        # Aggregate and print custom metric
        averaged_accuracy = sum(accuracies) / len(accuracies)
        print(f"Round {server_round} accuracy averaged: {averaged_accuracy}")
        return aggregated_loss, {"accuracy": averaged_accuracy}