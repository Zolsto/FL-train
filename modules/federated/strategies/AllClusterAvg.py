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
from torch.utils.tensorboard import SummaryWriter
from modules.federated.strategies.utils import get_evaluate_fn, get_fit_metrics_aggregation_fn


def get_group(cid, group_split: List[int]) -> str:
    '''
    Get the group index for a given client ID based on the group split.
    '''
    cid = int(cid)
    if group_split is None:
        return "group0"

    if cid < group_split[0]:
        return "group0"
    
    for i in range(1, len(group_split)):
        if group_split[i-1] <= cid < group_split[i]:
            return f"group{i}"
    
    raise ValueError(f"Client ID {cid} does not belong to any group in the split {group_split}.")
    return "group0"

class AllClusterAvg(Strategy):
    def __init__(
        self,
        group_split: List[int],
        param_split: int,
        group_at_end: bool = True,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        weighted_loss: bool = True,
        on_fit_config_fn: Optional[Callable] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        separate_eval: bool = False,
        initial_parameters: Optional[Parameters] = None,
        writer: Optional[SummaryWriter] = None,
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
        self.separate_eval = separate_eval
        self.group_split = group_split
        self.param_split = param_split
        self.group_at_end = group_at_end
        self.weighted_loss = weighted_loss
        self.writer = writer
        self.save_path = save_path
        if initial_parameters is not None:
            tensors = parameters_to_ndarrays(initial_parameters)
            if self.group_at_end:
                self.global_param = ndarrays_to_parameters(tensors[:self.param_split])
                group_initial = ndarrays_to_parameters(tensors[self.param_split:])
                self.group_param = {}
                for i in range(len(self.group_split)):
                    self.group_param[f"group{i}"] = group_initial
            else:
                self.global_param = ndarrays_to_parameters(tensors[self.param_split:])
                group_initial = ndarrays_to_parameters(tensors[:self.param_split])
                self.group_param = {}
                for i in range(len(self.group_split)):
                    self.group_param[f"group{i}"] = group_initial
        else:
            self.global_param = None
            self.group_param = { f"group{i}": None for i in range(len(self.group_split)) }


    def initialize_parameters(
        self,
        client_manager: ClientManager
        # INFO ClientManager
        # attributes: clients (Dict[str, ClientProxy])
        # methods: register, unregister, all, wait_for, sample, num_available
        ) -> Optional[Parameters]:
        """
        Initializes the global model parameters.
        If no initial parameters are provided, a new model is instantiated and
        its weights are used to initialize the global and group-specific parameters.

        Args:
            client_manager (ClientManager): The client manager.

        Returns:
            Optional[Parameters]: The initial model parameters to be sent to the clients.
        """
        
        # If no initial parameters are given, use base model to initialize
        if self.global_param is None:
            base_model = EfficientNetModel()
            state_dict = base_model.model.state_dict()
            all_parameters = [val.cpu().numpy() for val in state_dict.values()]
            if self.group_at_end:
                self.global_param = ndarrays_to_parameters(all_parameters[:self.param_split])
                for k, v in self.group_param.items():
                    if v is None:
                        self.group_param[k] = ndarrays_to_parameters(all_parameters[self.param_split:])
            else:
                self.global_param = ndarrays_to_parameters(all_parameters[self.param_split:])
                for k, v in self.group_param.items():
                    self.group_param[k] = ndarrays_to_parameters(all_parameters[:self.param_split])

            return ndarrays_to_parameters(all_parameters)
            
        global_param = parameters_to_ndarrays(self.global_param)
        group_param = parameters_to_ndarrays(self.group_param["group0"])
        all_parameters = global_param + group_param
        return ndarrays_to_parameters(all_parameters)
        
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        # INFO Parameters
        # attributes: tensor (List[bytes]), tensor_type (str)
        # methods: to_ndarrays, from_ndarrays (mostly conversion)
        client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configures the training instructions for a round.
        This method samples clients and constructs a personalized model for each
        by combining the current global parameters with the client's specific
        group parameters.

        Args:
            server_round (int): The current round of federated learning.
            parameters (Parameters): The current global model parameters.
            client_manager (ClientManager): The client manager.

        Returns:
            List[Tuple[ClientProxy, FitIns]]: A list of tuples, where each tuple
                contains a client proxy and the training instructions (FitIns)
                for that client.
        """

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
            properties_res = client.get_properties(ins=GetPropertiesIns(config={}), group_id=0, timeout=30)
            partition_id = int(properties_res.properties["partition_id"])
            group = get_group(partition_id, self.group_split)
            global_param = parameters_to_ndarrays(self.global_param)
            group_param = parameters_to_ndarrays(self.group_param[group])
            if self.group_at_end:
                parameters = ndarrays_to_parameters(global_param + group_param)
            else:
                parameters = ndarrays_to_parameters(group_param + global_param)

            print(f"Group {group[-1]}, partition ID {partition_id}")
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
        """
        Aggregates the training results from clients.
        This method separates the client updates into global and group-specific
        parts and aggregates them accordingly. The global part is averaged
        across all clients (with a fairness adjustment), while the group-specific
        parts are averaged only within their respective groups.

        Args:
            server_round (int): The current round of federated learning.
            results (List[Tuple[ClientProxy, FitRes]]): The successful results
                from the clients.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): The
                failures from the clients.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The updated global
                parameters and a metrics dictionary.
        """
        if not results:
            return None, {}
    
        # Initialize variables to save weights and num_examples
        updates = []
        
        base_param = parameters_to_ndarrays(results[0][1].parameters)
        global_avg = []
        group_avg = {}
        clients_trained = {}
        if self.group_at_end:
            global_avg = [np.zeros_like(layer, dtype=np.float32) for layer in base_param[:self.param_split]]
        else:
            global_avg = [np.zeros_like(layer, dtype=np.float32) for layer in base_param[self.param_split:]]

        # For each client
        for client, fit_res in results:
            # Get the weights, group and number of examples
            group = get_group(fit_res.metrics["partition_id"], self.group_split)
            weights = parameters_to_ndarrays(fit_res.parameters)
            # Save all in a tuple (0->group, 1->weight, 2->num_examples)
            train_data = (group, weights, fit_res.num_examples)
            updates.append(train_data)
            # initialize this group avg to 0
            if group not in group_avg.keys():
                clients_trained[group] = 0
                if self.group_at_end:
                    group_avg[group] = [np.zeros_like(layer, dtype=np.float32) for layer in base_param[self.param_split:]]
                else:
                    group_avg[group] = [np.zeros_like(layer, dtype=np.float32) for layer in base_param[:self.param_split]]

            clients_trained[group] += 1
        
        # Compute total number of examples (global and per group)
        group_examples = {}
        total_examples = 0
        for k in group_avg.keys():
            g_list = [item[2] for item in updates if item[0]==k]
            group_examples[k] = np.sum(g_list)
            total_examples += group_examples[k]
        
        # Compute global avg and per group avg (only on group present in training)
        for group, params, n in updates:
            global_w = 1 / (clients_trained[group]*len(self.group_split))
            group_w = n / group_examples[group]
            if self.group_at_end:
                global_update = params[:self.param_split]
                group_update = params[self.param_split:]
                for avg, update in zip(global_avg, global_update):
                    avg += update * global_w

                for avg, update in zip(group_avg[group], group_update):
                    avg += update * group_w
            else:
                group_update = params[:self.param_split]
                global_update = params[self.param_split:]
                for avg, update in zip(group_avg[group], group_update):
                    avg += update * group_w

                for avg, update in zip(global_avg, global_update):
                    avg += update * global_w
        
        # Update old parameters
        self.global_param = ndarrays_to_parameters(global_avg)
        for k, v in group_avg.items():
            self.group_param[k] = ndarrays_to_parameters(v)

        return self.global_param, {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configures the evaluation instructions for a round.
        Similar to `configure_fit`, this method constructs a personalized model
        for each client before sending evaluation instructions.

        Args:
            server_round (int): The current round of federated learning.
            parameters (Parameters): The current global model parameters.
            client_manager (ClientManager): The client manager.

        Returns:
            List[Tuple[ClientProxy, EvaluateIns]]: A list of tuples, where each tuple
                contains a client proxy and the evaluation instructions (EvaluateIns)
                for that client.
        """

        # Select clients to fit
        num_available = client_manager.num_available()
        sample_size = int(num_available * self.fraction_evaluate)
        num_clients = max(self.min_evaluate_clients, sample_size)
        clients = client_manager.sample(num_clients=num_clients, min_num_clients=self.min_evaluate_clients)
        # Get the evaluation config
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        tuples = []
        for client in clients:
            properties_res = client.get_properties(ins=GetPropertiesIns(config={}), group_id=0, timeout=30)
            partition_id = int(properties_res.properties["partition_id"])
            group = get_group(partition_id, self.group_split)
            global_param = parameters_to_ndarrays(self.global_param)
            group_param = parameters_to_ndarrays(self.group_param[group])
            if self.group_at_end:
                parameters = ndarrays_to_parameters(global_param + group_param)
            else:
                parameters = ndarrays_to_parameters(group_param + global_param)

            evaluate_ins = EvaluateIns(parameters, config)
            # INFO EvaluateIns is a tuple of (parameters, config), returned with its client (for each client)
            tuples.append((client, evaluate_ins))
            
        return tuples

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        # INFO: EvaluateRes is similar to FitRes, but for reporting test sessions
        # attributes: num_examples, loss, metrics (dict for other metrics)
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregates the evaluation results from clients.
        Calculates the weighted average loss and metrics for each group and
        for all clients combined.

        Args:
            server_round (int): The current round of federated learning.
            results (List[Tuple[ClientProxy, EvaluateRes]]): The successful
                evaluation results from the clients.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): The
                failures from the clients.

        Returns:
            Tuple[Optional[float], Dict[str, Scalar]]: A tuple containing the
                centralized loss and a dictionary of aggregated metrics.
        """

        if not results:
            return None, {}

        groups_metrics = { k : { "loss": [], "num_examples": [], "metrics": {} } for k in self.group_param.keys() }

        global_metrics = { "loss": [], "num_examples": [], "metrics": {} }

        for client, evaluate_res in results:
            group = get_group(evaluate_res.metrics["partition_id"], self.group_split)
        
            # Save group metrics
            groups_metrics[group]["loss"].append(evaluate_res.loss)
            groups_metrics[group]["num_examples"].append(evaluate_res.num_examples)
        
            # Save global metrics
            global_metrics["loss"].append(evaluate_res.loss)
            global_metrics["num_examples"].append(evaluate_res.num_examples)
        
            # Add other metrics if present
            for metric_name, metric_value in evaluate_res.metrics.items():
                if metric_name not in groups_metrics[group]["metrics"].keys():
                    groups_metrics[group]["metrics"][metric_name] = []

                if metric_name not in global_metrics["metrics"].keys():
                    global_metrics["metrics"][metric_name] = []

                groups_metrics[group]["metrics"][metric_name].append(metric_value)
                global_metrics["metrics"][metric_name].append(metric_value)
            
        # Aggregate all metrics
        aggregated_metrics = {}
        global_loss = np.average(global_metrics["loss"], weights=global_metrics["num_examples"])
        aggregated_metrics["global_loss"] = global_loss
        aggregated_metrics["global_examples"] = np.sum(global_metrics["num_examples"])
        for k, v in global_metrics["metrics"].items():
            weighted_avg = np.average(v, weights=global_metrics["num_examples"])
            aggregated_metrics[f"global_{k}"] = weighted_avg
            
        for group_name, group_data in groups_metrics.items():
            group_examples = np.sum(group_data["num_examples"])
            if group_examples > 0:
                # Group loss (weighted average)
                group_loss = np.average(group_data["loss"], weights=group_data["num_examples"])
                aggregated_metrics[f"{group_name}_loss"] = group_loss
                aggregated_metrics[f"{group_name}_examples"] = group_examples
            
                # Other metrics of the group (weighted average)
                for metric_name, metric_values in group_data["metrics"].items():
                    weighted_avg = np.average(metric_values, weights=group_data["num_examples"])
                    aggregated_metrics[f"{group_name}_{metric_name}"] = weighted_avg
            else:
                aggregated_metrics[f"{group_name}_loss"] = 0
                aggregated_metrics[f"{group_name}_examples"] = 0
                for metric in group_data["metrics"].keys():
                    aggregated_metrics[f"{group_name}_{metric}"] = 0

        if self.save_path is not None:
            for i in range(len(self.group_split)):
                group = f"group{i}"
                group_param = parameters_to_ndarrays(self.group_param[group])
                global_param = parameters_to_ndarrays(self.global_param)
                if self.group_at_end:
                    parameters = global_param + group_param
                else:
                    parameters = group_param + global_param
                
                model = EfficientNetModel()
                base_state_dict = model.state_dict()
                param_names = list(base_state_dict.keys())
                new_state_dict = {}
                for name, array in zip(param_names, parameters):
                    new_state_dict[name] = torch.from_numpy(array)

                model.load_state_dict(new_state_dict)
                torch.save(model.state_dict(), f"{self.save_path}/{group}.pt")

        return global_loss, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Performs a centralized evaluation on the server-side.
        This method iterates through each client group, reconstructs the
        appropriate model for that group, and runs evaluation using the
        provided `evaluate_fn`.

        Args:
            server_round (int): The current round of federated learning.
            parameters (Parameters): The current global model parameters.

        Returns:
            Optional[Tuple[float, Dict[str, Scalar]]]: A tuple containing the
                globally aggregated loss and a dictionary of all detailed metrics.
        """

        # Optional server evaluation
        if self.evaluate_fn is not None:
            #return self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
            all_results = {k : { "loss": 0.0, "accuracy": 0.0} for k in self.group_param.keys()}
            all_results["global"] = { "loss": 0.0, "accuracy": 0.0 }
            i = 0
            for group in self.group_param.keys():
                global_param = parameters_to_ndarrays(self.global_param)
                group_param = parameters_to_ndarrays(self.group_param[group])
                if self.group_at_end:
                    parameters = global_param + group_param
                else:
                    parameters = group_param + global_param

                group_loss, group_metric = self.evaluate_fn(server_round=server_round,
                    parameters=parameters,
                    config={},
                    name=group,
                    separate_eval=self.separate_eval)
                # Save the results for each group
                all_results[group]["loss"] = float(group_loss)
                all_results[group]["accuracy"] = float(group_metric["accuracy"])
                    
                # Weighted average for global results
                if self.weighted_loss:
                    if i == 0:
                        weight = self.group_split[i] / self.group_split[-1]
                    else:
                        weight = (self.group_split[i] - self.group_split[i-1]) / self.group_split[-1]
                
                else:
                    weight = 1 / len(self.group_param)

                all_results["global"]["loss"] += group_loss * weight
                all_results["global"]["accuracy"] += group_metric["accuracy"] * weight
                i += 1
            
            if self.writer:
                self.writer.add_scalar("global/loss", all_results["global"]["loss"], server_round)
                self.writer.add_scalar("global/accuracy", all_results["global"]["accuracy"], server_round)
            
            return all_results["global"]["loss"], all_results

        return None