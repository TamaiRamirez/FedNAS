"""flwr-enas-nofix: A Flower / TensorFlow app."""

import gc
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_enas_nofix.my_strategy import CustomFedAvg, CustomFedProxPerformace, CustomFedProx
from typing import List, Tuple

from flwr_enas_nofix.task import PROJECT_PATH, AGG_METHOD, MU, write_experiment_txt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
from tensorflow.keras import backend as K
import gc
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def fit_config(rnd: int):
    return {"round": rnd}


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
    else:
        strategy = CustomFedProxPerformace(
            num_rounds=num_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            on_fit_config_fn=fit_config,
            proximal_mu = MU
        )
    config = ServerConfig(num_rounds=num_rounds)
    K.clear_session()
    gc.collect()
    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
