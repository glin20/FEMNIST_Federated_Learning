from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from torchvision.models import resnet50, ResNet50_Weights
from src import train, test, load_datasets, get_parameters, set_parameters, DEVICE
import csv

# Determines whether the simulation runs with data poisoning
# Set to True to Poison 3 clients with 50% label flipping
# Set to False to have all clients have healthy data
POISONED = False

# Client implementation


class FlowerClient(NumPyClient):
    # Set client data to given parameters

    def __init__(self, net, trainloader, testloader, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.partition_id = partition_id

    # Write the evaluated loss and accuracy to csv file

    def write_data(self, loss, accuracy):
        with open("client_loss.csv", "a", newline='') as f_loss:
            writer = csv.writer(f_loss)
            writer.writerow([self.partition_id, loss])
        with open("client_acc.csv", "a", newline='') as f_acc:
            writer = csv.writer(f_acc)
            writer.writerow([self.partition_id, accuracy])

    # Get parameters of the model

    def get_parameters(self, config):
        return get_parameters(self.net)

    # Set model parameters to global model's and train the model
    # Return model's parameters

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader,
              self.partition_id, epochs=1, poisoned=POISONED)
        return get_parameters(self.net), len(self.trainloader), {}

    # Set parameters to global model's and evaluate the model
    # using the testloader to get the loss and accuracy and write
    # the data to csv file. Return the loss and accuracy to the server

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader,
                              self.partition_id, poisoned=POISONED)
        self.write_data(loss, accuracy)
        return float(loss), len(self.testloader), {"loss": float(loss), "accuracy": float(accuracy)}


# Loads the default model (ResNet50) along with its weights and gets
# the client's partition_id. Load the data partition based on
# partition_id and obtain the trainloader and testloader.
# Returns a client representation


def client_fn(context: Context) -> Client:

    # Load model
    net = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)

    # Get partition_id
    partition_id = context.node_config["partition-id"]

    # Load dataset partition based on partition_id and get
    # trainloader + testloader
    trainloader, testloader = load_datasets(
        partition_id=partition_id)

    return FlowerClient(net, trainloader, testloader, partition_id).to_client()


# Flower ClientApp
client_app = ClientApp(client_fn)
