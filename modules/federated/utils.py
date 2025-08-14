"""
This module contains utility functions for the federated learning process.

Classes:
    None

Functions:
    get_weights: Function to get the weights of a model.
    set_weights: Function to set the weights of a model.

Constants:
    None

Exceptions:
    None

Author: Matteo Caligiuri
"""

from typing import List
from collections import OrderedDict

import torch
import numpy as np


# Define the element to export
__all__ = [
    "get_weights",
    "set_weights",
]


def get_weights(net: torch.nn.Module) -> List[np.ndarray]:
    """
    Function to get the weights of a model.

    Args:
        net (torch.nn.Module): The model to get the weights from.

    Returns:
        List[np.ndarray]: The weights of the model.
    """

    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Function to set the weights of a model.

    Args:
        net (torch.nn.Module): The model to set the weights to.
        parameters (List[np.ndarray]): The weights to set.

    Returns:
        None
    """

    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
