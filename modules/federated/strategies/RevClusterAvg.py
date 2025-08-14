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

class RevClusterAvg(Strategy):
    def __init__(
        self,
        group_split: List[int],
        param_split: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        weighted_loss: bool = False,
        on_fit_config_fn: Optional[Callable] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        initial_parameters: Optional[Parameters] = None,
        writer: Optional[SummaryWriter] = None,
        ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        if self.fit_metrics_aggregation_fn is None:
            self.fit_metrics_aggregation_fn = get_fit_metrics_aggregation_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.evaluate_fn = evaluate_fn
        if self.evaluate_fn is None:
            self.evaluate_fn = get_evaluate_fn()
        self.group_split = group_split
        self.param_split = param_split
        self.weighted_loss = weighted_loss
        self.writer = writer
        if initial_parameters is not None:
            tensors = parameters_to_ndarrays(initial_parameters)
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
        '''
        Initialize the server parameters.
        '''
        
        # If no initial parameters are given, use base model to initialize
        if self.global_param is None:
            base_model = EfficientNetModel()
            state_dict = base_model.model.state_dict()
            all_parameters = [val.cpu().numpy() for val in state_dict.values()]
            self.global_param = ndarrays_to_parameters(all_parameters[self.param_split:])
            for k, v in self.group_param.items():
                self.group_param[k] = ndarrays_to_parameters(all_parameters[:self.param_split])

            return ndarrays_to_parameters(all_parameters)
            
        global_param = parameters_to_ndarrays(self.global_param)
        group_param = parameters_to_ndarrays(self.group_param["group1"])
        return ndarrays_to_parameters(group_param + global_param)
        
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
            properties_res = client.get_properties(ins=GetPropertiesIns(config={}), group_id=0, timeout=30)
            partition_id = int(properties_res.properties["partition_id"])
            group = get_group(partition_id, self.group_split)
            global_param = parameters_to_ndarrays(self.global_param)
            group_param = parameters_to_ndarrays(self.group_param[group])
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
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
    
        # Extract weights and number of examples of each clients
        global_updates = []
        group_updates = {}
        group_examples = {}
        num_examples = []

        base_params = parameters_to_ndarrays(results[0][1].parameters)
        group_avg = {}
        g = {}

        # For each client
        for client, fit_res in results:
            # Get the weights, group and number of examples
            weights = parameters_to_ndarrays(fit_res.parameters)
            group = get_group(fit_res.metrics["partition_id"], self.group_split)
            global_updates.append(weights[self.param_split:])
            if group not in group_updates:
                group_updates[group] = []
                group_avg[group] = [np.zeros_like(layer, dtype=np.float32) for layer in base_params[:self.param_split]]
                g[group] = 0

            group_updates[group].append(weights[:self.param_split])
            if group not in group_examples:
                group_examples[group] = 0

            group_examples[group] += fit_res.num_examples
            num_examples.append(fit_res.num_examples)
        
        total_examples = np.sum(num_examples)

        # Initialize averages to 0 (shape is taken form 1st client parameters)
        global_avg = []
        base_params = parameters_to_ndarrays(results[0][1].parameters)
        for layer in base_params[self.param_split:]:
            global_avg.append(np.zeros_like(layer, dtype=np.float32))
        
        #group_avg = {}
        #for k in group_updates.keys():
        #    group_avg[k] = [np.zeros_like(layer, dtype=np.float32) for layer in base_params[self.param_split:]]
        
        # Aggregate global and local parameters
        c = 0
        #g = { k : 0 for k in group_updates.keys() }
        for client, fit_res in results:
            group = get_group(fit_res.metrics["partition_id"], self.group_split)
            
            # Global updates are weighted equally across all clients: more clients = less weight
            global_weight = num_examples[c] / total_examples
            i = int(group[-1])
            if i==0:
                global_weight = global_weight / self.group_split[i]
            else:
                global_weight = global_weight / (self.group_split[i] - self.group_split[i-1])

            # INFO Each position of global_updates contains global weights of a client
            for layer in range(len(global_updates[0])):
                global_avg[layer] += global_updates[c][layer] * global_weight

            # Local updates are weighted by the number of examples in the group
            local_weight = num_examples[c] / group_examples[group]
            for layer in range(len(group_updates[group][0])):
                group_avg[group][layer] += group_updates[group][g[group]][layer] * local_weight
            
            g[group] += 1
            c += 1
        
        # Update old parameters
        self.global_param = ndarrays_to_parameters(global_avg)

        for k in group_avg.keys():
            self.group_param[k] = ndarrays_to_parameters(group_avg[k])

        global_w = parameters_to_ndarrays(self.global_param)
        group_w = parameters_to_ndarrays(self.group_param['group1'])
        new_parameters = ndarrays_to_parameters(group_w + global_w)
        return new_parameters, {}

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
            properties_res = client.get_properties(ins=GetPropertiesIns(config={}), group_id=0, timeout=30)
            partition_id = int(properties_res.properties["partition_id"])
            group = get_group(partition_id, self.group_split)
            global_param = parameters_to_ndarrays(self.global_param)
            group_param = parameters_to_ndarrays(self.group_param[group])
            parameters = ndarrays_to_parameters(group_param + global_param)
            evaluate_ins = EvaluateIns(parameters, config)
            # INFO EvaluateIns is a tuple of (parameters, config), returned with its client (for each client)
            tuples.append((client, evaluate_ins))

        return tuples

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        '''
        Aggregate evaluation results (weighted average).
        '''

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

        return global_loss, aggregated_metrics

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
            all_results = {k : { "loss": 0.0, "accuracy": 0.0} for k in self.group_param.keys()}
            all_results["global"] = { "loss": 0.0, "accuracy": 0.0 }
            i = 0
            for group in self.group_param.keys():
                global_param = parameters_to_ndarrays(self.global_param)
                group_param = parameters_to_ndarrays(self.group_param[group])
                parameters = group_param + global_param
                group_loss, group_metric = self.evaluate_fn(server_round, parameters, {}, name=group)
                # Save the results for each group
                all_results[group]["loss"] = group_loss
                all_results[group]["accuracy"] = group_metric["accuracy"]
                    
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