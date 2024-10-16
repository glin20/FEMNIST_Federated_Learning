from matplotlib import pyplot as plt
from collections import defaultdict
from flwr.simulation import run_simulation
from client import client_app
from server import server_app
from src import DEVICE
import os
import csv

NUM_CLIENTS = 10
BATCH_SIZE = 32

color = ["#828C8D", "#509E32", "#97D341", "#9F7B45", "#0C86FE",
         "#DAE276", "#39AE19", "#E358B7", "#F4734F", "#C3D8B2"]

files = ["client_loss.csv", "client_acc.csv",
         "average_loss.csv", "average_acc.csv"]

# Plots the data found in the files


def plot_client_data():
    for f in files:
        if (f.startswith("client")):
            dic = defaultdict(list)
            with open(f, 'r') as file:
                reader = csv.reader(file)
                for line in reader:
                    client, val = line[0], line[1]
                    dic[client].append(float(val))
                for var in dic:
                    plt.plot(dic[var], c=color[int(var)], marker=".")
                plt.savefig(f.split(".")[0])
                plt.show()
        else:
            arr = []
            with open(f, 'r') as file:
                reader = csv.reader(file)
                for line in file:
                    arr.append(float(line))
                plt.plot(arr, marker=".")
                plt.savefig(f.split(".")[0])
                plt.show()


# Remove file to allow for new file to be created
for f in files:
    if os.path.exists(f):
        os.remove(f)

backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

# Run simulation
run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)

# Plot data
plot_client_data()
