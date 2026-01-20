"""flwr-enas-nofix: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from .my_strategy import CustomFedAvg, CustomFedProxPerformace, CustomFedProx
from typing import List, Tuple

from flwr_enas_nofix.task import PROJECT_PATH, AGG_METHOD

def fit_config(rnd: int):
    return {"round": rnd}

# def weighted_average (metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     """Function to weight aggregate the metrics based on num examples"""
#     # Loop trough all metrics received compute accuracies x examples
#     accuracies = [num_examples * m["child_valid_acc"] for num_examples, m in metrics]
#     total_examples = sum(num_examples for num_examples, _ in metrics)
#     # Return weighted average accuracy
#     return {"weighted_val_accuracy": sum(accuracies) / total_examples}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    # parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    if AGG_METHOD == "fedavg":
        strategy = CustomFedAvg(
            num_rounds=num_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            
            on_fit_config_fn=fit_config
        )
    elif AGG_METHOD == "fedprox":
        # PENDIENTE DE IMPLEMENTAR
        None
    else:
        strategy = CustomFedProxPerformace(
            num_rounds=num_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            
            on_fit_config_fn=fit_config
        )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
