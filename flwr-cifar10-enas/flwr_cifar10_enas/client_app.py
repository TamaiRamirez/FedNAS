"""flwr-cifar10-enas: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flwr_cifar10_enas.task import load_data, load_model, train

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, data, parameters
    ):
        # self.model = model
        # self.x_train, self.y_train, self.x_test, self.y_test = data
        # self.epochs = epochs
        # self.batch_size = batch_size
        # self.verbose = verbose
        self.data = data
        self.parameters = parameters

    def fit(self, parameters, config):
        result = train(copy.deepcopy(self.data),self.parameters["architecture"],self.parameters["transfer"])
        return result

def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        data, parameters
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
