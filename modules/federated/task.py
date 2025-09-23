"""
This module contains the function of the training and evaluating loop
of the federated learning process.

Classes:
    None

Functions:
    train: Train the network on the training set.
    test: Evaluate the network on the entire test set.

Constants:
    None

Exceptions:
    None


Author: Matteo Caligiuri
"""

from typing import Union, Dict, Any, Optional

import torch


_all_ = ["train", "test"]


def train(
    model: torch.nn.Module,
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    epochs: int=1,
    fine_tune: bool=False,
    stop_norm: bool=False,
    bn_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train the network on the training set.

    Args:
        model (torch.nn.Module): The client model.
        device (Union[torch.device, str]): The device on which the training will be performed.
        loss (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        epochs (int): The number of epochs.

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # If it's a fine tune round, disable gradient for convolutional part of the model
    if fine_tune:
        for param in model.model.features.parameters():
            param.requires_grad = False

    # Cast the device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Define the losses and accuracies accumulators
    tot_loss, tot_acc = torch.zeros(epochs), torch.zeros(epochs)

    # Train the model
    for epoch in range(epochs):
        # Run an epoch
        metrics = _run_epoch(model, device, dataloader, epoch, criterion, optimizer, stop_norm=stop_norm, bn_index=bn_index)

        # Update the metrics
        tot_loss[epoch] = metrics["loss"]
        tot_acc[epoch] = metrics["accuracy"]

    # Compute the total loss and accuracy
    tot_loss = tot_loss.mean().item()
    tot_acc = tot_acc.mean().item()

    # Reactivate gradient of the convolutional part of the model
    if fine_tune:
        for param in model.model.features.parameters():
            param.requires_grad = True

    # Print the total metrics
    print(f"[Train results] - train {tot_loss:1.4f} | accuracy {(tot_acc * 100):6.2f}%")

    # Define the output metrics
    metrics = {"loss": tot_loss, "accuracy": tot_acc}

    return metrics


def test(
    model: torch.nn.Module,
    device: Union[torch.device, str],
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    bn_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate the network on the entire test set.

    Args:
        model (torch.nn.Module): The client model.
        device (Union[torch.device, str]): The device on which the training will be performed.
        criterion (torch.nn.Module): The loss function.
        dataloader (torch.utils.data.DataLoader): The test data loader.

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Cast the device if needed
    if isinstance(device, str):
        device = torch.device(device)
    # Evaluate the model
    metrics = _run_epoch(model, device, dataloader, 0, criterion, eval=True, bn_index=bn_index)

    # Extract the total loss and accuracy
    loss = metrics["loss"]
    acc = metrics["accuracy"]

    # Print the total metrics
    print(f"[Test results] - test loss {loss:1.4f} | accuracy {(acc * 100):6.2f}%")

    # Define the output metrics
    metrics = {"loss": loss, "accuracy": acc}

    return metrics


def _run_epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
    stop_norm: bool = False,
    bn_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run an epoch of training on the client model.

    Args:
        model (torch.nn.Module): The client model.
        device (str): The device on which the training will be performed.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        epoch (int): The epoch number.
        criterion (torch.nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer.
            If not in eval mode the optimizer is required.
            Default: None
        eval (bool): Whether to evaluate the model.
            Default: False

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Check if the input data are correct
    _in_checker(optimizer, eval)

    # Set the BatchNorm index if applicable
    if bn_index is not None:
        model.model.update_index(bn_index)

    # Set the model to the correct mode
    if eval:
        model.eval()
    elif stop_norm:
        model.train()
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    else:
        model.train()

    # Define the losses and accuracies accumulators
    epoch_loss, epoch_acc = torch.zeros(len(dataloader)), torch.zeros(len(dataloader))

    # Train the model
    for i, batch in enumerate(dataloader):
        # Run the batch
        metrics = _run_batch(model, device, batch, criterion, optimizer, eval)

        # Update the metrics
        epoch_loss[i] = metrics["loss"]
        epoch_acc[i] = metrics["accuracy"]

    # Compute the epoch loss and accuracy
    epoch_loss = epoch_loss.mean().item()
    epoch_acc = epoch_acc.mean().item()

    # Print the epoch metrics
    if not eval:
        print(f"Epoch {epoch+1}: loss {epoch_loss:1.4f} | accuracy {(epoch_acc * 100):6.2f}%")

    # Define the output metrics
    metrics = {"loss": epoch_loss, "accuracy": epoch_acc}

    return metrics


def _run_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eval: Optional[bool] = False,  # pylint: disable=redefined-builtin
) -> Dict[str, Any]:
    """
    Run a batch of training on the client model.

    Args:
        model (torch.nn.Module): The client model.
        device (str): The device on which the training will be performed.
        batch (Dict[str, torch.Tensor]): The batch of data.
        criterion (torch.nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer.
            If not in eval mode the optimizer is required.
            Default: None
        eval (bool): Whether to evaluate the model.
            Default: False

    Returns:
        Dict[str, Any]: The updated client model.
    """

    # Check if the input data are correct
    _in_checker(optimizer, eval)

    # Move the data to the device
    images, labels = batch[0], batch[1]
    images, labels = images.to(device), labels.to(device)

    # Compute the network prediction
    if not eval:
        optimizer.zero_grad()
    with torch.set_grad_enabled(not eval):
        outputs = model(images)["logits"]
        #print(outputs.shape, labels.shape)

    # Compute the loss
    loss = criterion(outputs, labels)

    if not eval:
        # Perform the backward pass
        loss.backward()
        optimizer.step()

    # Get the class predictions
    pred = torch.argmax(outputs, dim=1)

    # Compute the batch accuracy
    accuracy = torch.mean(1.0 * (pred == labels)).item()

    # Define the output metrics
    metrics = {"loss": loss.item(), "accuracy": accuracy}

    return metrics


def _in_checker(
    optimizer: torch.optim.Optimizer, eval: bool = False  # pylint: disable=redefined-builtin
) -> None:
    """
    Check if the input data are correct.
    if not in eval mode the loss function and the optimizer are required.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        eval (bool): Whether to evaluate the model.
            Default: False

    Returns:
        None
    """

    if not eval and optimizer is None:
        raise ValueError(
            "Since the network is in training mode, the optimizer is required."
        )

    return None
