"""flwr-enas-nofix: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ConfigRecord
import json

from flwr_enas_nofix.task import NUM_ROUNDS, load_data, Trainer, ndarray_to_weights, weights_to_ndarrays, save_controller_weights, load_controller_weights
import copy

from experiment_api import get_controller_weights_dir, get_results_dir, get_exp_id
import pandas as pd
import math

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, data, context: Context, partition_id
    ):
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

        if client_info["first_train"]:
            child_weights = None
            controller_weights = None
            transfer = False
        else:
            child_keys = client_info.get("child_weights_keys", None)
            child_weights = ndarray_to_weights(parameters, child_keys)
            controller_weights = load_controller_weights(get_controller_weights_dir() / f"client_{self.partition_id}_controller_weights.npz")
            transfer = True
        
        lr = 0.025 * 0.5 * (1 + math.cos(math.pi * (actual_round - 1) / (NUM_ROUNDS - 1)))
        
        trainer = Trainer(copy.deepcopy(self.data),{"child_weights": child_weights, "controller_trainable_variables": controller_weights}, transfer, actual_round, self.partition_id, learning_rate=lr)
        result = trainer.train()

        child_keys = list(result["child_weights"].keys())
        model_weights = weights_to_ndarrays(result["child_weights"], child_keys)
        
        client_info["last_valid_acc"] = result["child_valid_acc"]
        client_info["last_test_acc"] = result["child_test_acc"]
        client_info["last_valid_loss"] = result["child_valid_loss"]
        client_info["last_test_loss"] = result["child_test_loss"]
        client_info["last_valid_f1_score"] = result["child_valid_f1_score"]
        client_info["last_test_f1_score"] = result["child_test_f1_score"]
        client_info["child_weights_keys"] = child_keys
        save_controller_weights(result["controller_trainable_variables"], get_controller_weights_dir() / f"client_{self.partition_id}_controller_weights.npz")

        # print(f"[FIT] child_valid_acc={result["child_valid_acc"]}, child_test_acc={result["child_test_acc"]}")

        client_info["first_train"] = False

        results_client = {
            "exp_id": get_exp_id(),
            "round": actual_round,
            "client_id": self.partition_id,
            "val_accuracy": result["child_valid_acc"],
            "val_loss": result["child_valid_loss"],
            "val_f1_score": result["child_valid_f1_score"],
            "test_accuracy": result["child_test_acc"],
            "test_loss": result["child_test_loss"],
            "test_f1_score": result["child_test_f1_score"],
            "communication_cost": 2 * len(result["child_trainable_variables"]) * 4 / (1024 * 1024),
            "size_model": len(result["child_trainable_variables"]) * 4 / (1024 * 1024)
        }
        
        path = get_results_dir() / f"client_{self.partition_id}.parquet"
        df = pd.DataFrame([results_client])
        
        if path.exists():
            df_existing = pd.read_parquet(path)
            df = pd.concat([df_existing, df], ignore_index=True)

        df.to_parquet(path, engine="pyarrow", index=False)

        return model_weights, len(model_weights), {"valid_f1_score": result["child_valid_f1_score"]}
        # return {weights, size of weights, (results -> aggregate fit function)}
    
    def evaluate(self, parameters, config):
        client_info = self.client_state.config_records["client_info"]
        valid_acc = client_info.get("last_valid_acc", 0.0)
        test_acc = client_info.get("last_test_acc", 0.0)
        valid_loss = client_info.get("last_valid_loss", 0.0)
        test_loss = client_info.get("last_test_loss", 0.0)
        test_f1_score = client_info.get("last_test_f1_score", 0.0)
        valid_f1_score = client_info.get("last_valid_f1_score", 0.0)
        return test_loss, len(self.data["images"]["test"]), {
            "child_valid_acc": valid_acc,
            "child_valid_loss": valid_loss,
            "child_valid_f1_score": valid_f1_score,
            "child_test_acc": test_acc,
            "child_test_loss": test_loss,
            "child_test_f1_score": test_f1_score
        }

def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    return FlowerClient(
        data, context, partition_id
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
