"""
This module contains all the elements needed to create a Flower Client instance

Classes:
    FlowerClient: Flower client implementation for PyTorch.

Functions:
    generate_client_fn: Generate the client_fn function to use for the federated learning process.

Constants:
    None

Exceptions:
    None


Author: Matteo Caligiuri
"""

from typing import Union, Dict

import torch
from flwr.client import NumPyClient
from flwr.common import Context
from omegaconf import DictConfig
from hydra.utils import instantiate

from .task import train, test
from .utils import get_weights, set_weights

import os
import logging


# Define wahat to export
__all__ = ["generate_client_fn"]


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    """
    Flower client implementation for PyTorch.

    Args:
        partition_id (int): The partition ID.
        model (torch.nn.Module): The model to train.
        device (Union[torch.device, str]): The device to use for the training.
        criterion (torch.nn.Module): The loss function.
        optimizer (Union[torch.optim.Optimizer, DictConfig]): The optimizer to use for the training.
        trainloader (torch.utils.data.DataLoader): The training data loader.
        valloader (torch.utils.data.DataLoader): The validation data loader
    """

    def __init__(
        self,
        partition_id: int,
        model: torch.nn.Module,
        device: Union[torch.device, str],
        criterion: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, DictConfig],
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        save_path: str = None
    ):

        # Initialize the FlowerClient variables
        self.partition_id = partition_id
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.save_path = save_path

        # Logging
        os.makedirs(self.save_path, exist_ok=True)

        # Ogni client scriverà in un file diverso (es. client_0.log, client_1.log)
        log_file_path = os.path.join(self.save_path, f"client_{self.partition_id}.log")

        # Configura il logger per scrivere su file
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format=f'%(asctime)s - Client {self.partition_id} - %(levelname)s - %(message)s',
            force=True)

        #logging.info("Logger inizializzato.")

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_weights(self.model)

    def get_properties(self, config: Dict={}, group_id: int=0):
        """
        Get the partition ID of the client.

        Returns:
            int: The partition ID.
        """
        properties = {"partition_id": self.partition_id}
        return properties

    def fit(self, parameters, config):
        # Print client info
        print(f"[Client {self.partition_id}] fit, config: {config}")

        # Set the model parameters
        set_weights(self.model, parameters)

        # Define the optimizer
        if isinstance(self.optimizer, DictConfig):
            optimizer = instantiate(self.optimizer, lr=config["lr"], params=self.model.parameters())
        else:
            optimizer = self.optimizer(params=self.model.parameters(), lr=1e-4)

        # Train the model
        metrics = train(
            model=self.model,
            device=self.device,
            criterion=self.criterion,
            optimizer=optimizer,
            dataloader=self.trainloader,
            epochs=config["local_epochs"],
        )
        metrics["partition_id"] = self.partition_id
        return get_weights(self.model), len(self.trainloader), metrics

    def evaluate(self, parameters, config):
        # Print client info
        print(f"[Client {self.partition_id}] evaluate, config: {config}")

        # Set the model parameters
        set_weights(self.model, parameters)

        # Evaluate the model
        metrics = test(
            model=self.model,
            device=self.device,
            criterion=self.criterion,
            dataloader=self.valloader,
        )
        metrics["partition_id"] = self.partition_id
        logging.info(f"Evaluation metrics: {metrics}")
        # Extract the loss from the metrics
        loss = metrics.pop("loss")
        return float(loss), len(self.valloader), metrics


def generate_client_fn(
    model: torch.nn.Module,
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, DictConfig],
    dataloaders: callable,
    save_path: str = None
) -> callable:
    """
    Generate the client_fn function to use for the federated learning process.

    Args:
        model (torch.nn.Module): The model to train.
        device (Union[torch.device, str]): The device to use for the training.
        criterion (torch.nn.Module): The loss function.
        optimizer (Union[torch.optim.Optimizer, DictConfig]): The optimizer to use for the training.
        dataloaders (callable): The function used to load the data

    Returns:
        callable: The client_fn function tor use for the federated learning process.
    """

    def client_fn(context: Context) -> FlowerClient:
        """
        Client function used to create a Flower Client instance.

        Args:
            context (flwr.common.Context): The context used to create the Flower Client instance.

        Returns:
            FlowerClient: The Flower Client instance.
        """

        # Load model and data
        partition_id = context.node_config["partition-id"]
        data = dataloaders(partition_id)

        # Return Client instance
        return FlowerClient(
            partition_id=partition_id,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            trainloader=data["train"],
            valloader=data["val"],
            save_path=save_path
        ).to_client()

    return client_fn
