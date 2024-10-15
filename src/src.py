from collections import OrderedDict
from typing import List

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.utils.logging import disable_progress_bar

import flwr
from flwr_datasets import FederatedDataset
from datasets import load_dataset
from flwr_datasets.partitioner import NaturalIdPartitioner

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

NUM_CLIENTS = 10
BATCH_SIZE = 32

# Code taken from Flower Framework's Getting Started with Flower
# https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
# Edited dataset to be FEMNIST dataset and changed the transforms images to three channels
# as FEMNIST is grayscale rather than RGB. Batch keys also changed to match the FEMNIST dataset.


def load_datasets(partition_id: int):
    # Load local FEMNIST subset dataset
    dataset_dict = load_dataset(
        "imagefolder", data_dir=f"./femnist_subset/client_{partition_id}")
    dataset = dataset_dict["train"]

    # Divide data on each node: 80% train, 20% test
    partition_train_test = dataset.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(),
         # "Convert" grayscale image to RGB
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(
        apply_transforms)

    testloader = DataLoader(
        partition_train_test["test"], batch_size=BATCH_SIZE)

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    return trainloader, testloader


def poison(batch_labels):
    labels = []
    for label in batch_labels:
        if label.item() in [0, 1, 2, 3, 4]:
            labels.append((label.item() + 1) % 5)
        else:
            labels.append(label.item())
    return torch.tensor(labels).to(DEVICE)


def train(net, trainloader, partition_id, epochs: int, poisoned):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(
                DEVICE), batch["label"].to(DEVICE)
            if partition_id % 4 == 0 and poisoned:
                labels = poison(labels)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(
                DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
