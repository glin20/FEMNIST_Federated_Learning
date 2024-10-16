from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from torchvision.models import resnet50, ResNet50_Weights
from src import train, test, load_datasets, get_parameters, set_parameters, DEVICE

# Determines whether the simulation runs with data poisoning
# 1 = Regular poison, half of partitions shifted up 1 label
# 2 = Random Poison, half of all labels randomly reassigned
# 3 = All Threes Poison, partitions 0,1,2,4,5 relabeled to 3
# Any other number = No Poison
POISONED = False


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id

    def write_data(self, loss, accuracy):
        with open("client_loss.data", "a") as f_loss:
            f_loss.write(f"{self.partition_id},{loss}\n")
        with open("client_acc.data", "a") as f_acc:
            f_acc.write(f"{self.partition_id},{accuracy}\n")

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader,
              self.partition_id, epochs=1, poisoned=POISONED)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.partition_id, poisoned=POISONED)
        self.write_data(loss, accuracy)
        return float(loss), len(self.valloader), {"loss": float(loss), "accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)

    partition_id = context.node_config["partition-id"]
    trainloader, valloader = load_datasets(
        partition_id=partition_id)

    return FlowerClient(net, trainloader, valloader, partition_id).to_client()


# Flower ClientApp
client_app = ClientApp(client_fn)
