from matplotlib import pyplot as plt, colors
import matplotlib.cm as cm
from collections import defaultdict
from flwr.simulation import run_simulation
from client import client_app
from server import server_app
from src import DEVICE
import os

NUM_CLIENTS = 10
BATCH_SIZE = 32

color = ["#828C8D", "#509E32", "#97D341", "#9F7B45", "#0C86FE",
         "#DAE276", "#39AE19", "#E358B7", "#F4734F", "#C3D8B2"]
norm = colors.Normalize(vmin=0, vmax=NUM_CLIENTS)

files = ["client_loss.data", "client_acc.data",
         "average_loss.data", "average_acc.data"]


def plot_client_data():
    for f in files:
        if (f.startswith("client")):
            dic = defaultdict(list)
            with open(f, 'r') as f_loss:
                for line in f_loss:
                    client, val = line.split(",")
                    dic[client].append(float(val))
                for var in dic:
                    plt.plot(dic[var], c=color[int(var)], marker=".")
                plt.savefig(f.split(".")[0])
                plt.show()
        else:
            arr = []
            with open(f, 'r') as f_loss:
                for line in f_loss:
                    arr.append(float(line))
                plt.plot(arr, marker=".")
                plt.savefig(f.split(".")[0])
                plt.show()


for f in files:
    if os.path.exists(f):
        os.remove(f)

backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)

plot_client_data()
