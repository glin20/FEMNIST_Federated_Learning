from typing import List, Tuple
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import csv
from logging import WARNING
import numpy as np
import math
from typing import Callable, Optional, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import awkward as ak
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

reputation_values = [None] * 10
# Calculate the average loss and accuracy metric given by the clients
# and write them to a csv file

#Metaparameters
#Determines number of rounds of aggregation occur before reputation and trust is applied
rounds_before_trust = 5
#Determines threshold trust value for removal from aggregation
beta = 0.2

class FedAvgEdit(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[
            int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.reputations = [None] * 10
        self.unregister = []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        print(self.unregister)
        for i in self.unregister:
            client_manager.unregister(i)
            self.min_available_clients -= 1
            self.min_fit_clients -= 1
            self.min_evaluate_clients -= 1
        self.unregister = []
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def get_Kmean_center(self, weights_result):
        flat = []
        for weights in weights_result:
            flat.append(ak.ravel(weights[0]))

        kmeans = KMeans(n_clusters=1).fit(flat)
        center = kmeans.cluster_centers_[0]
        return center
        # return ak.ravel(center)

    def get_normalized_distances(self, center, weights_results):
        distances = []
        for i in range(len(weights_results)):
            ping = ak.ravel(weights_results[i][0])
            d = np.linalg.norm(
                center - ak.ravel(weights_results[i][0]))
            distances.append([d])
        scaler = MinMaxScaler()
        scaler.fit(distances)
        normalized_distances = scaler.transform(distances)
        return normalized_distances.flatten()


    def get_reputation(self, normalized_distances, server_round):
        # d is set to 1 - normalized_dist as otherwise server will eventually discard all clients for some reason
        for i in range(len(normalized_distances)):
            d = 1- (normalized_distances[i])
            if server_round == 1:
                r = (1.0 - d)
            else:
                if d < (1.0 - d):
                    k = (self.reputations[i] + d) - (self.reputations[i] / server_round)
                    r = min(1, max(0, k))
                else:
                    k = (self.reputations[i] + d) - (np.exp(-(1.0 - (d * (self.reputations[i] / server_round)))))
                    r = min(1, max(0, k))
            self.reputations[i] = r

    def get_trust(self, normalized_distances):
        trusts = []
        # d is set to 1 - normalized_dist as otherwise server will eventually discard all clients for some reason
        for i in range(len(self.reputations)):
            d = 1- normalized_distances[i]
            trust = np.sqrt(self.reputations[i]**2 + d ** 2) - np.sqrt((1.0-self.reputations[i]) ** 2 + ((1.0-d) ** 2))
            trust = min(1, max(0, trust))
            trusts.append(trust)
        return trusts

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}


        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, client)
            for client, fit_res in results
        ]
        # Before Here Exactly the same as regular FedAvg

        aggregated_ndarrays = self.get_Kmean_center(weights_results)
        # raveled_aggregated = ak.ravel(aggregated_ndarrays)
        raveled_aggregated = aggregated_ndarrays # Ravel is unnecessary as resultant is already a flattened 1d array
        print(raveled_aggregated)

        normalized_distances = self.get_normalized_distances(
            raveled_aggregated, weights_results)
        print("NORM DISTANCE: ", normalized_distances)
        self.get_reputation(
            normalized_distances, server_round)
        print("REPUTATION: ", self.reputations)

        if server_round < rounds_before_trust:
            aggregated_results = aggregate_inplace(results)
        else:
            indexes = []
            index = []
            trust_values = self.get_trust(normalized_distances)
            print("TRUST: ", trust_values)
            trusted_clients = []
            for i in range(len(trust_values)):
                if trust_values[i] > beta:
                    trusted_clients.append(weights_results[i][:2])
                else:
                    indexes.append(self.reputations[i])
                    self.unregister.append(weights_results[i][2])
                    index.append(i)
            aggregated_results = aggregate(trusted_clients)
            for i in indexes:
                self.reputations.remove(i)
            print(index)
        parameters_aggregated = ndarrays_to_parameters(aggregated_results)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics)
                           for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
# --------------- End FedAvgEdit ----------------------------------------------

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    with open("average_loss.csv", "a", newline='') as f_loss:
        writer = csv.writer(f_loss)
        writer.writerow([sum(losses) / sum(examples)])
    with open("average_acc.csv", "a", newline='') as f_acc:
        writer = csv.writer(f_acc)
        writer.writerow([sum(accuracies) / sum(examples)])

    return {"accuracy": sum(accuracies) / sum(examples)}

# Server implementation using FedAvg strategy running for
# default 10 rounds.


def server_fn(context: Context) -> ServerAppComponents:
    strategy = FedAvgEdit(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Number of rounds process runs for
    config = ServerConfig(num_rounds=10)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
server_app = ServerApp(server_fn=server_fn)
