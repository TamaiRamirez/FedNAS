"""flwr-enas-nofix: A Flower / TensorFlow app."""

import gc
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_enas_nofix.my_strategy import CustomFedAvg, CustomFedProxPerformace, CustomFedProx, CustomQFedAvg
from typing import List, Tuple

import math

from flwr_enas_nofix.task import PROJECT_PATH, AGG_METHOD, MU, write_experiment_txt, ABLATION, NUM_ROUNDS, Q_PARAM

import tensorflow as tf

# def fit_config(rnd: int):
#     return {"round": rnd}

# Modificacion para Q-FedAvg
def fit_config(rnd: int):
    config = {"round": rnd}

    if AGG_METHOD == "qfedavg":
        total_rounds_lr = 40 if ABLATION else NUM_ROUNDS

        lr = 0.05 * 0.5 * (
            1 + math.cos(
                math.pi * (rnd - 1) / (total_rounds_lr - 1)
            )
        )

        config["q_lr"] = float(lr)

    return config



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    # parameters = ndarrays_to_parameters(load_model().get_weights())
    
    # Create experiment metadata txt
    write_experiment_txt()
    
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
        strategy = CustomFedProx(
            num_rounds=num_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            on_fit_config_fn=fit_config,
            proximal_mu = MU
        )
    elif AGG_METHOD == "custom":
        strategy = CustomFedProxPerformace(
            num_rounds=num_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            on_fit_config_fn=fit_config,
            proximal_mu = MU
        )
    elif AGG_METHOD == "qfedavg":

        strategy = CustomQFedAvg(
            num_rounds=num_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            on_fit_config_fn=fit_config,
            q_param=Q_PARAM,
            qffl_learning_rate=0.05,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
