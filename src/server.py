from typing import List, Tuple
from src import DEVICE, load_datasets, set_parameters, test
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib as plt


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    with open("average_loss.data", "a") as f_loss:
        f_loss.write(f"{sum(losses) / sum(examples)}\n")
    with open("average_acc.data", "a") as f_acc:
        f_acc.write(f"{sum(accuracies) / sum(examples)}\n")

    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=10)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
server_app = ServerApp(server_fn=server_fn)
