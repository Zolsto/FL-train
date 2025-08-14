"""
Module to handle the datasets.

Classes:
    _FedDataLoader: Load the federated datasets.

Functions:
    load_fed_dts: Load the datasets and build the dataloaders for the federated learning.
    _build_dataloaders: Build the dataloaders for the datasets.
"""

from typing import Optional, Dict, Union

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from modules.federated.partitioners import CombinedDirichletPartitioner

from modules.common import Logger, TqdmLogger
from modules.federated.partitioners import Partitioner


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()
tqdml = TqdmLogger(logger)


__all__ = ["load_fed_dts"]


def load_fed_dts(
    batch_size: int,
    partitioner: Dict[str, CombinedDirichletPartitioner],
    generator: Optional[torch.Generator] = None,
) -> callable:
    """
    Load the datasets and build the dataloaders.

    Args:
        batch_size (int): The batch size.
        seed (Optional[int], optional): The seed to use for the partitioning. Defaults to None.
        generator (Optional[torch.Generator], optional): The random generator.
            Defaults to None.
        partitioner (Dict[str, CombinedDirichletPartitioner]): A dict containing the partitions for both train and val.

    Returns:
        callable: The function to get the dataloader for a specific partition id.
    """

    # Build the dataloaders object
    return _FedDataLoader(
        partitioner=partitioner,
        batch_size=batch_size,
        num_workers=4,
        generator=generator,
    )


def _build_dataloaders(
    dts,
    batch_size: int,
    num_workers: Optional[int] = 4,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, DataLoader]:
    """
    Build the dataloaders for the datasets.

    Args:
        dts (DatasetDict): The datasets.
        batch_size (int): The batch size.
        num_workers (Optional[int], optional): The number of workers to use. Defaults to 4.
        generator (Optional[torch.Generator], optional): The random generator. Defaults to None.
        dts_type (Optional[str], optional): The dataset type. Defaults to "torch".
    Returns:
        Dict[str, DataLoader]: The dataloaders.
    """

    dataloaders = {}
    for k, v in dts.items():
        if isinstance(v, Dataset):
            dataloaders[k] = DataLoader(
                v,
                batch_size=batch_size,
                num_workers=num_workers,
                generator=generator,
                shuffle=True,
                drop_last=True,
            )
        elif isinstance(v, Partitioner):
            dataloaders[k] = DataLoader(
                v,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
            )
        elif isinstance(v, list):
            dataloaders[k] = [
                DataLoader(
                    i,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    generator=generator,
                    shuffle=True,
                    drop_last=True,
                )
                for i in v
            ]
        else:
            raise ValueError(f"Dataset data type not supported: {type(v)}")

    return dataloaders

class _FedDataLoader:
    """
    This class is responsible for loading the federated datasets.
    given a partitioner object for each split (as a dict) when called with a specific id
    returns the dataloader for that partitioner.
    When called with a specific id, it returns the dataloader for that partition.

    Args:
        partitioner (Dict[str, CombinedDirichletPartitioner]): A dict containing the partitions for both train and val.
        batch_size (int): The batch size.
        num_workers (Optional[int], optional): The number of workers to use. Defaults to 4.
        generator (Optional[torch.Generator], optional): The random generator. Defaults to None.
    """

    def __init__(
        self,
        partitioner: Dict[str, CombinedDirichletPartitioner],
        batch_size: int,
        num_workers: Optional[int] = 4,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.partitioners = partitioner
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = generator

    def __call__(
        self, id: Union[int, str]  # pylint: disable=redefined-builtin
    ) -> DataLoader:
        """
        Call the object with a specific id to get the dataloader for that partition.

        Args:
            id (Union[int, str]): The partition id.
        Returns:
            DataLoader: The dataloader for the partition.
        """

        # Parse the id
        if isinstance(id, str):
            id = int(id)

        return _build_dataloaders(
            dts={
                k: v.load_partition(partition_id=id)
                for k, v in self.partitioners.items()
            },
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.generator,
        )
