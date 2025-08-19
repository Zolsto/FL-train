"""
Module for utility functions used by the federated strategies.

Classes:
    None

Functions:
    set_strategy: Set the strategy to use for the federated learning process.
    get_on_fit_config: Get the function to configure the server's training process.
    get_fit_metrics_aggregation_fn: Get the function to aggregate the training metrics.
    get_evaluate_fn: Get the function to evaluate the model on the test set.
    get_on_evaluate_config: Get the function to configure the server's evaluation process.

Constants:
    None

Exceptions:
    None

"""

from typing import Union, Optional, Callable, List, Dict, Tuple
from pathlib import Path
from collections import OrderedDict
from importlib import import_module
from logging import INFO


import numpy as np
import torch
import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
)
from flwr.common.logger import log
from omegaconf import DictConfig

from modules.federated.task import test
from modules.federated.utils import set_weights

from torch.utils.tensorboard import SummaryWriter


# Define the element to export
__all__ = [
    "set_strategy",
    "get_on_fit_config",
    "get_fit_metrics_aggregation_fn",
    "get_evaluate_fn",
    "get_on_evaluate_config",
]


def set_strategy(
    strategy: str,
    net: Optional[torch.nn.Module] = None,
    dts_name: Optional[str] = None,
    save_every: Optional[int] = 1,
    save_path: Optional[Union[str, Path]] = None,
) -> Callable:
    """
    Set the strategy to use for the federated learning process, ensuring that the
    aggregate_fit method is customized to save the model after aggregation.

    Args:
        strategy (str): Full import path to the strategy class
            (e.g., "flwr.server.strategy.FedAvg").
        net (Optional[torch.nn.Module], optional): The model to save. Defaults to None.
        dts_name (Optional[str], optional): Name of the dataset. Defaults to None.
            Used to name the saved model.
        save_every (int, optional): Number of rounds before saving the model. Defaults to 1.
        save_path (Union[str, Path], optional): Path to save the model. Defaults to None.

    Returns:
        Callable: The customized strategy class.
    """

    # Load the strategy class dynamically
    strategy_class = _load_strategy(strategy)

    # Convert save_path to Path object if it's a string
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
    elif save_path is None or net is None:
        # No customization needed
        return strategy_class

    # Define the custom aggregate_fit method
    # Call the base class method and save the model after aggregation
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        # Call base class method
        aggregated_parameters, aggregated_metrics = strategy_class.aggregate_fit(
            self, server_round, results, failures
        )

        if (
            aggregated_parameters is not None
            and server_round % save_every == 0
            and save_every > 0
        ):
            log(
                INFO, "[SERVER]: Saving round %s aggregated_parameters...", server_round
            )

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model to disk
            if dts_name is not None:
                ckpt_name = (
                    f"{server_round:03d}_{net.__class__.__name__.lower()}"
                    + f"_{dts_name.lower()}_federated_checkpoint.pt"
                )
            else:
                ckpt_name = (
                    f"{server_round:03d}_{net.__class__.__name__.lower()}"
                    + "_federated_checkpoint.pt"
                )
            ckpt_path = save_path / ckpt_name
            torch.save(net.state_dict(), ckpt_path)

            log(INFO, "[SERVER]: Model %s saved.", ckpt_path.name)

        return aggregated_parameters, aggregated_metrics

    # Create the custom strategy class dynamically
    CustomStrategy = type(  # pylint: disable=invalid-name
        f"{strategy_class.__name__}",
        (strategy_class,),
        {"aggregate_fit": aggregate_fit},
    )

    return CustomStrategy


def _load_strategy(strategy: str) -> Strategy:
    """
    Import a strategy dynamically from its full module path.

    Args:
        strategy (str): Full path of the strategy (e.g., "flwr.server.strategy.FedAvg").

    Returns:
        object: The imported strategy class or function.
    """

    module_name = ".".join(strategy.split(".")[:-1])
    class_name = strategy.split(".")[-1]

    # Import the module dynamically
    module = import_module(module_name)

    # Get the class or function from the module
    return getattr(module, class_name)


def get_on_fit_config(config: DictConfig) -> Callable[[int], Dict]:
    """
    Get the function to configure the server's training process.

    Args:
        config (DictConfig): The configuration object.

    Returns:
        Callable[int, Dict]: The function to configure the server's training process.
    """

    # Parse the configuration to a dict
    try:
        config = dict(config)
    except TypeError:
        config = {}

    def fit_config_fn(server_round: int) -> Dict:

        return {
            **config,
            "server_round": server_round,
        }

    return fit_config_fn


def get_on_evaluate_config(config: DictConfig) -> Callable[[int], Dict]:
    """
    Get the function to configure the server's evaluation process.

    Args:
        config (DictConfig): The configuration object.

    Returns:
        Callable[int, Dict]: The function to configure the server's training process.
    """

    # Parse the configuration to a dict
    try:
        config = dict(config)
    except TypeError:
        config = {}

    def evaluate_config_fn(server_round: int) -> Dict:

        return {
            **config,
            "server_round": server_round,
        }

    return evaluate_config_fn


def get_fit_metrics_aggregation_fn() -> Callable[[List[Tuple[int, Dict]]], Dict]:
    """
    Get the function to aggregate the training metrics.

    Args:
        None

    Returns:
        Callable[[List[Tuple[int, Dict]]], Dict]: The function to aggregate the training metrics.
    """

    def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:

        # Initialize the aggregated metrics
        aggregated_metrics = {k: 0.0 for k in metrics[0][1].keys()}

        # Aggregate the metrics
        count = 0
        for n, metric in metrics:
            for k, v in metric.items():
                aggregated_metrics[k] += v * n
                count += n

        for k in aggregated_metrics.keys():
            aggregated_metrics[k] /= count

        return aggregated_metrics

    return fit_metrics_aggregation_fn


def get_evaluate_fn(
    model: torch.nn.Module,
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    writer: Optional[SummaryWriter] = None,
    loader_dict: Optional[Dict[str, torch.utils.data.DataLoader]] = None
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]:
    """
    Get the function to evaluate the model on the test set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (Union[torch.device, str]): The device on which the evaluation will be performed.
        dataloader (torch.utils.data.DataLoader): The test data loader.

    Returns:
        Callable[int, NDArrays, Dict[str, Scalar], Tuple[float, Dict[str, Scalar]]]:
            The function to evaluate the model on the test set.
    """

    # Parse the device
    if isinstance(device, str):
        device = torch.device(device)

    # Define the evaluation function
    def evaluate_fn(
        server_round: int,  # pylint: disable=unused-argument
        parameters: NDArrays,
        config: Dict[str, Scalar],  # pylint: disable=unused-argument
        name: str="global",
        separate_eval: bool=False
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update the model with the latest parameters
        set_weights(model, parameters)

        if name!="global" and loader_dict is not None and separate_eval:
            # Use the specific dataloader for the group
            test_loader = loader_dict[name]
            if server_round==1:
                print(f"Using separate evaluation for {name} with {len(test_loader.dataset)} samples.")
        else:
            # Use the default dataloader
            test_loader = dataloader
            if server_round==1:
                print(f"Using global evaluation with {len(test_loader.dataset)} samples.")

        # Evaluate the model
        metrics = test(
            model=model,
            device=device,
            criterion=criterion,  # config["criterion"],
            dataloader=test_loader,
        )
        if writer is not None:
            #print(f"Logging {name} results: loss({metrics['loss']}) accuracy({metrics['accuracy']})")
            writer.add_scalar(f"{name}/loss", metrics['loss'], server_round)
            writer.add_scalar(f"{name}/accuracy", metrics['accuracy'], server_round)
            
        return metrics["loss"], {"accuracy": metrics["accuracy"]}

    return evaluate_fn
