"""flwr-cifar10-enas: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ConfigRecord

from flwr_cifar10_enas.task import PROJECT_PATH, load_data, Trainer, ndarray_to_weights, weights_to_ndarrays, save_controller_weights, load_controller_weights
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
        self.partition_id = partition_id
        self.child_valid_acc = None
        self.child_test_acc = None
        # Initialize context to know if it is the first time the model is used or not
        self.client_state = context.state
        if "client_info" not in self.client_state.config_records:
            self.client_state.config_records["client_info"] = ConfigRecord({"first_train": True})

    def fit(self, parameters, config):
        actual_round = config["round"]
        client_info = self.client_state.config_records["client_info"] # Charge context

        # After the first round we have to use the generic model for 12 classes 
        if client_info["first_train"]:
            child_weights = None
            controller_weights = None
            transfer = False
        else:
            child_keys = client_info.get("child_weights_keys", None)
            child_weights = ndarray_to_weights(parameters, child_keys)
            controller_weights = load_controller_weights(f"{PROJECT_PATH}/client_{self.partition_id}_controller_weights.pkl")
            transfer = True
        trainer = Trainer(copy.deepcopy(self.data),{"child_weights": child_weights, "controller_trainable_variables": controller_weights}, transfer, actual_round, self.partition_id)
        result = trainer.train()

        child_keys = list(result["child_weights"].keys())
        model_weights = weights_to_ndarrays(result["child_weights"], child_keys)
        
        client_info["last_valid_acc"] = result["child_valid_acc"]
        client_info["last_test_acc"] = result["child_test_acc"]
        client_info["last_valid_loss"] = result["child_valid_loss"]
        client_info["last_test_loss"] = result["child_test_loss"]
        client_info["child_weights_keys"] = child_keys
        save_controller_weights(result["controller_trainable_variables"], f"{PROJECT_PATH}/client_{self.partition_id}_controller_weights.pkl")

        # print(f"[FIT] child_valid_acc={result["child_valid_acc"]}, child_test_acc={result["child_test_acc"]}")

        client_info["first_train"] = False
        return model_weights, len(model_weights), {}
    
    def evaluate(self, parameters, config):
        client_info = self.client_state.config_records["client_info"]
        valid_acc = client_info.get("last_valid_acc", 0.0)
        test_acc = client_info.get("last_test_acc", 0.0)
        valid_loss = client_info.get("last_valid_loss", 0.0)
        test_loss = client_info.get("last_test_loss", 0.0)
        # print(f"[EVALUATE] child_valid_acc={valid_acc}, child_test_acc={test_acc}")
        return test_loss, len(self.data["images"]["test"]), {
            "child_valid_acc": valid_acc,
            "child_valid_loss": valid_loss,
            "child_test_acc": test_acc,
            "child_test_loss": test_loss

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
