"""flwr-cifar10-enas: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from .my_strategy import CustomFedAvg

def fit_config(rnd: int):
    return {"round": rnd}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    # parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = CustomFedAvg(
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
