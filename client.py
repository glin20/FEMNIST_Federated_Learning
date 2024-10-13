from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from torchvision.models import resnet50, ResNet50_Weights

from src import train, test, load_datasets, get_parameters, set_parameters, DEVICE


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        print("loss: {loss}, accuracy: {accuracy}", loss, accuracy)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)

    partition_id = context.node_config["partition-id"]
    trainloader, valloader = load_datasets(partition_id=partition_id)

    return FlowerClient(net, trainloader, valloader).to_client()


# Flower ClientApp
client_app = ClientApp(client_fn)
