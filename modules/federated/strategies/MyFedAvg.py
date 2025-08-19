from flwr.server.strategy import Strategy
from flwr.common import (
    FitRes,
    EvaluateRes,
    FitIns,
    EvaluateIns,
    Parameters,
    Scalar,
    NDArrays,
    GetPropertiesIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from typing import List, Optional, Tuple, Dict, Callable, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
import numpy as np
import flwr
import torch
from modules.federated.efficientnet import EfficientNetModel
from modules.federated.strategies.utils import get_evaluate_fn, get_fit_metrics_aggregation_fn

class MyFedAvg(Strategy):
    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            on_fit_config_fn: Optional[Callable] = None,
            on_evaluate_config_fn: Optional[Callable] = None,
            evaluate_fn: Optional[Callable] = None,
            fit_metrics_aggregation_fn: Optional[Callable] = None,
            initial_parameters: Optional[Parameters] = None,
            save_path: Optional[str] = None
        ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.evaluate_fn = evaluate_fn
        self.best_loss = np.inf
        self.parameters = initial_parameters
        self.save_path = save_path

    def initialize_parameters(
        self,
        client_manager: ClientManager
        # INFO ClientManager
        # attributes: clients (Dict[str, ClientProxy])
        # methods: register, unregister, all, wait_for, sample, num_available
        ) -> Optional[Parameters]:
        '''
        Initialize the server parameters.
        '''
        
        # If no initial parameters are given, use base model to initialize
        if self.parameters is None:
            base_model = EfficientNetModel()
            state_dict = base_model.model.state_dict()
            parameters = [val.cpu().numpy() for val in state_dict.values()]
            parameters = ndarrays_to_parameters(parameters)
            self.parameters = parameters

        return self.parameters
        
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        # INFO Parameters
        # attributes: tensor (List[bytes]), tensor_type (str)
        # methods: to_ndarrays, from_ndarrays (mostly conversion)
        client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
        '''
        Select clients and prepare fit (training) instructions.
        '''

        # Select clients to fit
        num_available = client_manager.num_available()
        sample_size = int(num_available * self.fraction_fit)
        num_clients = max(self.min_fit_clients, sample_size)
        clients = client_manager.sample(num_clients=num_clients, min_num_clients=self.min_fit_clients)
        # Get the fit config (same for all clients)
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        # Each client needs its group parameters
        tuples = []
        print(f"In round {server_round} selected {len(clients)} clients for fitting:")
        for client in clients:
            fit_ins = FitIns(parameters, config)
            # INFO FitIns is a tuple of (parameters, config), returned with its client (for each client)
            tuples.append((client, fit_ins))
        print()
        return tuples

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        # INFO: FitRes is used to bring result from training
        # attributes: parameters (trained weights of the client), num_examples (training examples of the client), num_examples_ceil, fit_duration
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        examples = []
        weights = []

        for _, fitres in results:
            examples.append(fitres.num_examples)
            weights.append(parameters_to_ndarrays(fitres.parameters))
        
        total_examples = np.sum(examples)
        base_param = parameters_to_ndarrays(results[0][1].parameters)
        avg = [ np.zeros_like(layer, dtype=np.float32) for layer in base_param ]
        for client in range(len(weights)):
            w = examples[client] / total_examples
            for i in range(len(weights[client])):
                avg[i] += weights[client][i] * w
        
        self.parameters = ndarrays_to_parameters(avg)
        return self.parameters, {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        '''
        Select clients and prepare evaluation instructions.
        '''

        # Select clients to fit
        num_available = client_manager.num_available()
        sample_size = int(num_available * self.fraction_evaluate)
        num_clients = max(self.min_evaluate_clients, sample_size)
        clients = client_manager.sample(num_clients=num_clients, min_num_clients=self.min_evaluate_clients)
        # Get the evaluation config
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        tuples = []
        for client in clients:
            evaluate_ins = EvaluateIns(parameters, config)
            # INFO EvaluateIns is a tuple of (parameters, config), returned with its client (for each client)
            tuples.append((client, evaluate_ins))
            
        return tuples

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        # INFO: EvaluateRes is similar to FitRes, but for reporting test sessions
        # attributes num_examples, loss, metrics (dict for other metrics)
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        '''
        Aggregate evaluation results (weighted average).
        '''

        if not results:
            return None, {}
        
        all_metrics = { "loss": [], "num_examples": [] }
        for _, evres in results:
            all_metrics['loss'].append(evres.loss)
            all_metrics['num_examples'].append(evres.num_examples)
            for metric_name, metric_value in evres.metrics.items():
                if metric_name not in all_metrics.keys():
                    all_metrics[metric_name] = []

                all_metrics[metric_name].append(metric_value)
        
        aggregated_metrics = {}
        for metric_name, metric_values in all_metrics.items():
            if metric_name=="num_examples":
                aggregated_metrics[metric_name] = np.sum(metric_values)
            else:
                aggregated_metrics[metric_name] = np.average(metric_values, weights=all_metrics['num_examples'])
        
        if self.fraction_evaluate == 0:
            self.save_model(server_round)

        return aggregated_metrics['loss'], aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        '''
        Evaluate the model on the server.
        '''

        # Optional server evaluation
        if self.evaluate_fn is not None:
            loss, metrics = self.evaluate_fn(server_round=server_round,
                parameters=parameters_to_ndarrays(parameters),
                config={})
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_model(server_round)

            return loss, metrics

        return None

    def save_model(self, server_round: int) -> None:
        """
        Save the current model parameters to disk,
        if a path was provided.

        Args:
            round (int): The current round of federated learning.
        """
        if self.save_path is not None:
            parameters = parameters_to_ndarrays(self.parameters) 
            model = EfficientNetModel()
            base_state_dict = model.state_dict()
            param_names = list(base_state_dict.keys())
            new_state_dict = {}
            for name, array in zip(param_names, parameters):
                new_state_dict[name] = torch.from_numpy(array)

            model.load_state_dict(new_state_dict)
            torch.save(model.state_dict(), f"{self.save_path}/model.pt")
            print(f"Model saved at round {server_round}.")