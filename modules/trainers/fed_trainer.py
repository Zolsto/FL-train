from typing import OrderedDict, Union, Optional, Tuple, Dict
from pathlib import Path

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.errors import InstantiationException
from torchvision.transforms.v2 import Compose
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.server.strategy import Strategy
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters

from modules.dataset.data_handler import load_fed_dts
from modules.federated.partitioners import CombinedDirichletPartitioner
from modules.common.logger import Logger
from modules.federated.client_app import generate_client_fn
from modules.federated.server_app import generate_server_fn
from modules.federated.utils import get_weights
from modules.federated.visualization import plot_label_distributions
from modules.federated.strategies.utils import set_strategy, get_evaluate_fn


# Define the logger to use
logger = Logger(name=__name__, level="info").get_logger()


# Define the element to export
__all__ = ["FedTrainer"]


class FedTrainer:
    """
    A class responsible for training a federated learning model.

    Args:
        c_model (Union[DictConfig, torch.nn.Module]): The classification model to use.
        s_model (Union[DictConfig, torch.nn.Module]): The segmentation model to use.
        n_classes (int): The number of classes in the dataset.
        criterion (Union[DictConfig, torch.nn.Module]): The criterion to use for the training.
        optimizer (Union[DictConfig, torch.optim.Optimizer]): The optimizer to use for the training.
        device (Optional[Union[str, torch.device]]): The device to use for the training.
            Defaults to "cpu".
        save_path (Optional[Union[Path, str]]): The path where to save the outputs.
            Defaults to Path("fed_outputs").
        checkpoint (Optional[Union[Path, str, OrderedDict]]): The checkpoint to load.
            Defaults to None.

    Returns:
        None
    """

    def __init__(
        self,
        c_model: torch.nn.Module,
        n_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[Union[str, torch.device]] = "cpu",
        save_path: Optional[Union[Path, str]] = Path("fed_outputs"),
        checkpoint: Optional[Union[Path, str, OrderedDict]] = None,
    ) -> None:
        
        if isinstance(save_path, str):
            save_path = Path(save_path)

        self.save_path = save_path

        # Set the device
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        if isinstance(c_model, torch.nn.Module):
            self.c_model = c_model
        else:
            self.c_model = instantiate(
                c_model,
                num_classes=n_classes,
                out_feats_size=n_classes,
            ).to(device)

        # Handle the checkpoint
        self._load_checkpoint(checkpoint)

        # Instantiate the criterion and the optimizer
        # if they are DictConfig objects
        if isinstance(criterion, DictConfig):
            criterion = instantiate(criterion)

        # Set the criterion and the optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        # Initialize the client
        self.client = None

        # Initialize the server
        self.server = None

        # Initialize the dataloaders
        self.dataloaders = None
        self.full_test_dataloader = None

    def _load_checkpoint(
        self, checkpoint: Optional[Union[Path, str, OrderedDict]] = None
    ) -> None:
        """
        Load the pretraining checkpoint.

        Args:
            checkpoint (Optional[Union[Path, str, OrderedDict]]): The checkpoint to load.

        Returns:
            None
        """

        # If checkpoint is None, return
        if checkpoint is None:
            logger.info(
                "No checkpoint provided. The client model will be initialized on "
                + "the ImageNet weights."
            )
            return

        # Process the checkpoint
        if isinstance(checkpoint, str):
            checkpoint = Path(checkpoint)
        elif isinstance(checkpoint, Path):
            checkpoint = torch.load(checkpoint)
        elif isinstance(checkpoint, OrderedDict):
            pass
        else:
            raise ValueError(
                "The checkpoint must be a string, a Path object, or an OrderedDict."
            )

        # Load the checkpoint
        checkpoint = torch.load(checkpoint)

        # Load the model
        self.c_model.load_state_dict(checkpoint["model"], strict=False)

    def set_data(
        self,
        dataloader_test: torch.utils.data.DataLoader,
        batch_size: int,
        partitioner: Dict[str, CombinedDirichletPartitioner],
        seed: Optional[int] = 42,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Set the data to use for the federated learning training and evaluation.

        Args:
            dataloader_test (torch.utils.data.DataLoader): The dataloader for the test set.
            batch_size (int): The batch size to use.
            partitioner (Dict[str, CombinedDirichletPartitioner]): A dict containing the partitions for both train and val.
            seed (Optional[int]): The seed to use for the data.
                Defaults to 42.
            generator (Optional[torch.Generator]): The generator to use for the data.
                If the generator is not provided, the seed will be used.
                Defaults to None.

        Returns:
            None
        """

        # Define the genearator if the generator is not provided
        # but the seed is provided
        if generator is None and seed is not None:
            generator = torch.Generator().manual_seed(seed)

        # Load the dataloaders of the dataset
        logger.info("Loading cliets dataset...")
        self.dataloaders = load_fed_dts(
            batch_size=batch_size,
            partitioner=partitioner,
            generator=generator,
        )
        logger.info("Dataset loaded.")
        logger.info("Loading the evaluation dataset...")
        self.full_test_dataloader = dataloader_test

        logger.info("Evaluation dataset loaded.")

        # Create all the partiotions
        # to speed up the training
        logger.info("Creating partitions...")
        for k in self.dataloaders.partitioners:
            self.dataloaders.partitioners[k].create_partitions()
        logger.info("Partitions created.")

    def set_client_and_server(
        self,
        strategy: Union[DictConfig, Strategy],
        num_rounds: int,
        log_every: Optional[int] = -1,
    ) -> None:
        """
        Set the client and the server for the federated learning process.

        Args:
            strategy (Union[DictConfig, Strategy]): The strategy to use for the
                federated learning process
            num_rounds (int): The number of rounds to run the federated learning process
            log_every (Optional[int]): The number of rounds after which to log the results.

        Returns:
            None
        """

        # Generate the client_fn function
        client_fn = generate_client_fn(
            model=self.c_model,
            device=self.device,
            criterion=self.criterion,
            optimizer=self.optimizer,
            dataloaders=self.dataloaders,
        )

        # Set the client
        self.client = ClientApp(client_fn)

        # Initialize the strategy if it is a DictConfig
        if isinstance(strategy, DictConfig):
            # Get the weights of the model
            params = get_weights(self.c_model)

            # Convert to dict the DictConfig
            # This is necessary to instantiate only the child elements
            strategy = dict(strategy)

            # Define the save path for the checkpoints
            ckpt_save_path = self.save_path / "fed_checkpoints"
            ckpt_save_path.mkdir(parents=True, exist_ok=True)

            # Modify the strategy to save the model
            strategy_class = set_strategy(
                strategy=strategy.pop("_target_"),
                net=self.c_model,
                dts_name=self.dataloaders.partitioners[
                    "train"
                ].dataset.__class__.__name__,
                save_every=log_every,
                save_path=ckpt_save_path,
            )

            # Instantiate all the child parameters
            for key, value in strategy.items():
                try:
                    strategy[key] = instantiate(value)
                except InstantiationException:
                    strategy[key] = value

            # Define the evaluate function
            evaluate_fn = get_evaluate_fn(
                model=self.c_model,
                device=self.device,
                criterion=self.criterion,
                dataloader=self.full_test_dataloader,
            )

            # Instantiate the modified strategy
            strategy = strategy_class(
                **strategy,
                initial_parameters=ndarrays_to_parameters(params),
                evaluate_fn=evaluate_fn,
            )

        # Generate the server_fn function
        server_fn = generate_server_fn(strategy=strategy, num_rounds=num_rounds)

        # Set the server
        self.server = ServerApp(server_fn=server_fn)

    def __call__(
        self, num_clients: int, log_clients_print: Optional[bool] = False
    ) -> None:
        """
        Train the model on the training set.

        Args:
            num_clients (int): The number of clients to use.
            log_clients_print (Optional[bool]): Whether to log the clients print.
                Defaults to False.

        Returns:
            None
        """

        backend_config = {"client_resources": None}
        if self.device.type == "cuda":
            backend_config = {
                "client_resources": {"num_cpus": 4, "num_gpus": 1},
                "actor": "torch",
                "init_args": {
                    "logging_level": 30,
                    "log_to_driver": log_clients_print,  # Allow each client to print to console
                },
            }

        run_simulation(
            client_app=self.client,
            server_app=self.server,
            num_supernodes=num_clients,
            backend_config=backend_config,
        )

    def plot_partitions(
        self,
        partitioner: Optional[str] = "train",
        custom_save_path: Optional[Union[str, Path]] = None,
        cmap: Optional[str] = None,
        bar_plot_size: Optional[Tuple[int]] = None,
        heatmap_size: Optional[Tuple[int]] = None,
    ) -> None:
        """
        Plot the parameters of the model.

        Args:
            partitioner (Optional[str]): The partitioner to use.
                Defaults to "train".
            custom_save_path (Optional[Union[str, Path]]): The path where to save the plot.
                Defaults to None. If None, the plot will be saved in the data path proided
                in the constructor.
            cmap (Optional[str]): The colormap to use for the plot.
                Defaults to None.
            bar_plot_size (Optional[Tuple[int]]): The size of the bar plot.
                Defaults to None.
            heatmap_size (Optional[Tuple[int]]): The size of the heatmap.
                Defaults to None.

        Returns:
            None
        """

        # Get the correct partitioner
        partitioner = self.dataloaders.partitioners[partitioner]
        partitioner_type = partitioner.__class__.__name__
        dataset_name = partitioner.dataset.__class__.__name__

        # Defien the save_path
        if custom_save_path is not None:
            # Check if the custom_save_path is a string
            # and convert it to a Path object
            if isinstance(custom_save_path, str):
                save_path = Path(custom_save_path)
            else:
                save_path = custom_save_path
        else:
            save_path = (
                self.save_path / f"partitions_plots/{partitioner_type}/{dataset_name}"
            )

        # Create the save_path if it does not exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Get the plot of the label distribution
        logger.info("Plotting the label distribution...")
        abs_bar_plot, _, df = plot_label_distributions(
            data=self.dataloaders.partitioners["train"],
            label_name=1,
            cmap=cmap,
            figsize=bar_plot_size,
            tight_layout=True,
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=False,
            verbose_labels=True,
            legend_kwargs={"ncol": 20},
            title="Per Partition Labels Distribution",
        )
        # Save the plot
        abs_bar_plot.savefig(save_path / "abs_bar_plot.pdf")
        logger.info("Absolute bar plot created.")

        perc_bar_plot, _, _ = plot_label_distributions(
            data=df,
            label_name=1,
            cmap=cmap,
            figsize=bar_plot_size,
            tight_layout=True,
            plot_type="bar",
            size_unit="percent",
            partition_id_axis="x",
            legend=False,
            verbose_labels=True,
            legend_kwargs={"ncol": 20},
            title="Per Partition Labels Distribution",
        )
        # Save the plot
        perc_bar_plot.savefig(save_path / "perc_bar_plot.pdf")
        logger.info("Percentage bar plot created.")

        heatmap, _, _ = plot_label_distributions(
            data=df,
            label_name=1,
            figsize=heatmap_size,
            tight_layout=True,
            plot_type="heatmap",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Per Partition Labels Distribution",
        )
        # Save the plot
        heatmap.savefig(save_path / "heatmap.pdf")
        logger.info("Heatmap created.")
        logger.info("Label distribution plots created.")

    def evaluate_on_test_set(self):
        self.c_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.full_test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.c_model(data)
                logits = output["logits"]  # Extract logits from the output dictionary
                test_loss += self.criterion(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.full_test_dataloader.dataset)
        accuracy = 100. * correct / len(self.full_test_dataloader.dataset)
        return {'test_loss': test_loss, 'accuracy': accuracy}
