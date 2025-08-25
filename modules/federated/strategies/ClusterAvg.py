import flwr
import torch
import numpy as np

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
from modules.federated.efficientnet import EfficientNetModel
from torch.utils.tensorboard import SummaryWriter
#from modules.federated.strategies.utils import get_evaluate_fn, get_fit_metrics_aggregation_fn
from modules.federated.utils import get_weights, set_weights

class MoreClusterAvg(Strategy):
    def __init__(
        self,
        # list of partition IDs at the start of each group
        # [0, index1, ..., indexN, num_clients]
        # group1 IDs (index1 --> index2 - 1), N+1 groups (last is groupN)
        group_split: List[int],
        # list of layers: True -> group, False -> global
        param_split: List[bool],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        # Whether to weight loss of a group by its size
        weighted_loss: bool = True,
        # Whether to test a group only on its own test set
        separate_eval: bool = True,
        on_fit_config_fn: Optional[Callable] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        initial_parameters: Optional[Parameters] = None,
        # TensorBoard writer
        writer: Optional[SummaryWriter] = None,
        # Path to save the model
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
        self.weighted_loss = weighted_loss
        self.writer = writer
        self.save_path = save_path
        self.group_param = { f"group{i}": [] for i in range(len(self.group_split)) }
        self.best_loss = { f"group{i}": np.inf for i in range(len(self.group_split)) }
        
        if initial_parameters is not None:
            for group in self.group_param.keys():
                self.global_param, self.group_param[group] = self.split_model(initial_parameters)
            self.global_param = ndarrays_to_parameters(self.global_param)
            for group in self.group_param.keys():
                self.group_param[group] = ndarrays_to_parameters(self.group_param[group])
        else:
            self.global_param = None


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
            initial_parameters = [val.cpu().numpy() for val in state_dict.values()]
            self.global_param, cluster_param = self.split_model(initial_parameters)
            self.global_param = ndarrays_to_parameters(self.global_param)
            for group in self.group_param.keys():
                self.group_param[group] = ndarrays_to_parameters(cluster_param)

            return ndarrays_to_parameters(initial_parameters)
    
        return ndarrays_to_parameters(self.assemble_model("group0"))
        
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
            group = self.get_group(partition_id)
            parameters = ndarrays_to_parameters(self.assemble_model(group))
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
        group_avg = {}
        global_avg = [np.zeros_like(layer, dtype=np.float32) for layer in parameters_to_ndarrays(self.global_param)]

        # For each client
        for client, fit_res in results:
            # Get the weights, group and number of examples
            group = self.get_group(fit_res.metrics["partition_id"])
            weights = parameters_to_ndarrays(fit_res.parameters)
            # Save all in a tuple (0->group, 1->weight, 2->num_examples)
            train_data = (group, weights, fit_res.num_examples)
            updates.append(train_data)
            # initialize this group avg to 0
            if group not in group_avg.keys():
                group_avg[group] = [np.zeros_like(layer, dtype=np.float32) for layer in parameters_to_ndarrays(self.group_param[group])]
        
        # Compute total number of examples (global and per group)
        group_examples = {}
        total_examples = 0
        for k in group_avg.keys():
            g_list = [item[2] for item in updates if item[0]==k]
            group_examples[k] = np.sum(g_list)
            total_examples += group_examples[k]
        
        # Compute global avg and per group avg (only on group present in training)
        base_w = 1 / len(self.group_param)
        for group, params, n in updates:
            group_w = n / group_examples[group]
            global_w = group_w * base_w
            global_update, group_update = self.split_model(params)
            for avg, update in zip(global_avg, global_update):
                avg += update * global_w

            for avg, update in zip(group_avg[group], group_update):
                avg += update * group_w
        
        # If some groups were not trained, their average is considered as unchanged (abstain)
        if len(group_examples) < len(self.group_param):
            old_param = parameters_to_ndarrays(self.global_param)
            for group in self.group_param.keys():
                # If this group was not trained...
                if group not in group_examples.keys():
                    # consider its average as unchanged
                    for avg, old in zip(global_avg, old_param):
                        avg += old * base_w
                    
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
            group = self.get_group(partition_id)
            parameters = ndarrays_to_parameters(self.assemble_model(group))
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

        groups_metrics = { k : { "loss": [], "num_examples": [] } for k in self.group_param.keys() }

        global_metrics = { "loss": [], "num_examples": [] }

        for client, evaluate_res in results:
            group = self.get_group(evaluate_res.metrics["partition_id"])
        
            # Save group metrics
            groups_metrics[group]["loss"].append(evaluate_res.loss)
            groups_metrics[group]["num_examples"].append(evaluate_res.num_examples)
        
            # Save global metrics
            global_metrics["loss"].append(evaluate_res.loss)
            global_metrics["num_examples"].append(evaluate_res.num_examples)
        
            # Add other metrics if present
            for metric_name, metric_value in evaluate_res.metrics.items():
                if metric_name not in groups_metrics[group].keys():
                    groups_metrics[group][metric_name] = []

                if metric_name not in global_metrics.keys():
                    global_metrics[metric_name] = []

                groups_metrics[group][metric_name].append(metric_value)
                global_metrics[metric_name].append(metric_value)
            
        # Aggregate all metrics
        aggregated_metrics = {}
        for k, v in global_metrics.items():
            if k=="num_examples":
                weighted_avg = np.sum(v)
            else:
                weighted_avg = np.average(v, weights=global_metrics["num_examples"])

            aggregated_metrics[f"global_{k}"] = weighted_avg
            
        for group_name, group_data in groups_metrics.items():
            group_examples = np.sum(group_data["num_examples"])
            if group_examples > 0:
                # Metrics of the group (weighted average)
                for metric_name, metric_values in group_data.items():
                    if metric_name=="num_examples":
                        weighted_avg = group_examples
                    else:
                        weighted_avg = np.average(metric_values, weights=group_data["num_examples"])

                    aggregated_metrics[f"{group_name}_{metric_name}"] = weighted_avg
            else:
                for metric in group_data.keys():
                    aggregated_metrics[f"{group_name}_{metric}"] = 0

        if self.fraction_evaluate == 0:
            self.save_model(server_round)

        return aggregated_metrics["global_loss"], aggregated_metrics

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
                parameters = self.assemble_model(group)
                group_loss, group_metric = self.evaluate_fn(server_round=server_round,
                    parameters=parameters,
                    config={},
                    name=group,
                    separate_eval=self.separate_eval)
                # Save the results for each group
                all_results[group]["loss"] = group_loss
                all_results[group]["accuracy"] = group_metric["accuracy"]

                if group_loss<self.best_loss[group]:
                    self.best_loss[group] = group_loss
                    self.save_model(server_round=server_round, group=group)
                    
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
    
    def assemble_model(self, group: str) -> NDArrays:
        """
        Assemble the model parameters from the global and group-specific parameters.
        This method reconstructs the model state dictionary from the current global
        and group-specific parameters.

        Args:
            group (str): The group name to assemble the model for. If "all", it assembles
                the model for all groups.

        Returns:
            NDArrays: parameters of the model as array.
        """
        if group not in self.group_param.keys():
            raise ValueError(f"Group {group} not found in {list(self.group_param.keys())}.")
            return None
        
        param = []
        global_param = parameters_to_ndarrays(self.global_param)
        group_param = parameters_to_ndarrays(self.group_param[group])
        c = 0
        g = 0
        for l in range(len(self.param_split)):
            if self.param_split[l]:
                # If this is a group layer, take it from groups parameters
                param.append(group_param[g])
                g += 1
            else:
                # If this is not a group layer, take it from global parameters
                param.append(global_param[c])
                c += 1

        return param

    def split_model(self, parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        """
        Split the model parameters into global and group-specific parts.
        This method separates the provided parameters into global parameters
        and group-specific parameters based on the defined parameter split.

        Args:
            parameters (NDArrays): The model parameters to split.

        Returns:
            Tuple[NDArrays, NDArrays]: A tuple containing global
                parameters and group-specific parameters.
        """
        global_param = []
        group_param = []
        if isinstance(parameters, Parameters):
            parameters = parameters_to_ndarrays(parameters)

        for l in range(len(self.param_split)):
            if self.param_split[l]:
                # If this is a group layer, save it in the groups parameters
                group_param.append(parameters[l])
            else:
                # If this is not a group layer, save it in the global parameters
                global_param.append(parameters[l])

        return global_param, group_param
            

    def save_model(self, server_round: int, group: str="all") -> None:
        """
        Save the current model parameters to disk.
        This method saves the global and group-specific parameters to the specified
        save path, if provided.

        Args:
            round (int): The current round of federated learning.
        """
        if self.save_path is not None:
            if group == "all":
                for k in self.group_param.keys():
                    self.save_model(server_round=server_round, group=k)
            else:        
                parameters = self.assemble_model(group)
                model = EfficientNetModel()
                set_weights(model, parameters)
                torch.save(model.state_dict(), f"{self.save_path}/{group}.pt")
                print(f"Model for {group} saved at round {server_round}.")
        else:
            return None

    def get_group(self, cid: int) -> str:
        '''
        Get the group index for a given client ID based on the group split.
        '''
        cid = int(cid)
        if self.group_split is None:
            return "group0"

        if cid < self.group_split[0]:
            return "group0"
    
        for i in range(1, len(self.group_split)):
            if self.group_split[i-1] <= cid < self.group_split[i]:
                return f"group{i}"
        
        raise ValueError(f"Client ID {cid} does not belong to any group in the split {self.group_split}.")
        return "group0"