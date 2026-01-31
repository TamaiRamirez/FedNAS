"""flwr-enas-nofix: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_enas_nofix.my_strategy import CustomFedAvg, CustomFedProxPerformace, CustomFedProx
from typing import List, Tuple

from flwr_enas_nofix.task import PROJECT_PATH, AGG_METHOD, MU, write_experiment_txt

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

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
