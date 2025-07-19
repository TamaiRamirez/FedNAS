"""flwr-cifar10-enas: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ConfigRecord

from flwr_cifar10_enas.task import load_data, Trainer, ndarray_to_weights, weights_to_ndarrays
import copy

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, data, context: Context, partition_id
    ):
        # self.model = model
        # self.x_train, self.y_train, self.x_test, self.y_test = data
        # self.epochs = epochs
        # self.batch_size = batch_size
        # self.verbose = verbose
        self.data = data
        self.keys = None
        self.controller_weights = None
        self.partition_id = partition_id
        self.child_valid_acc = None
        self.child_test_acc = None
        # Initialize context to know if it is the first time the model is used or not
        self.client_state = context.state
        if "client_info" not in self.client_state.config_records:
            self.client_state.config_records["client_info"] = ConfigRecord({"first_train": True, "last_train_save_dir": ""})

    def fit(self, parameters, config):
        actual_round = config["round"]
        client_info = self.client_state.config_records["client_info"] # Charge context

        # After the first round we have to use the generic model for 12 classes 
        if client_info["first_train"]:
            child_weights=None
            transfer=False
        else:
            child_weights=ndarray_to_weights(parameters,self.keys)
            transfer=True
        trainer = Trainer(copy.deepcopy(self.data),{"child_weights": child_weights, "controller_trainable_variables": self.controller_weights}, transfer, actual_round, self.partition_id)
        result = trainer.train()

        self.keys = list(result["child_weights"].keys())
        model_weights = weights_to_ndarrays(result["child_weights"], self.keys)
        self.controller_weights = result["controller_trainable_variables"]
        
        client_info = self.client_state.config_records["client_info"]
        client_info["last_valid_acc"] = result["child_valid_acc"]
        client_info["last_test_acc"] = result["child_test_acc"]
        # print(f"[FIT] child_valid_acc={result["child_valid_acc"]}, child_test_acc={result["child_test_acc"]}")
        return model_weights, len(model_weights), {}
    
    def evaluate(self, parameters, config):
        client_info = self.client_state.config_records["client_info"]
        valid_acc = client_info.get("last_valid_acc", 0.0)
        test_acc = client_info.get("last_test_acc", 0.0)
        # print(f"[EVALUATE] child_valid_acc={valid_acc}, child_test_acc={test_acc}")
        return 0.0, len(self.data["images"]["test"]), {
            "child_valid_acc": valid_acc,
            "child_test_acc": test_acc,
        }

def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    #data = load_data(partition_id, num_partitions)
    data = load_data(0, 1) # No partitions
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    # Return Client instance
    return FlowerClient(
        data, context, partition_id
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
