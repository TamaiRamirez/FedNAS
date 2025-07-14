"""Flwr-Cifar10-basic: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from flwr_cifar10_basic.task import load_data, load_model, PROJECT_PATH


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose, partition_id
    ):
        self.model = model
        self.images, self.labels = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.actual_round = 0
        self.partition_id = partition_id

    @staticmethod # Por defecto al ser un metodo de clase se le pasa la variable self pero entonces se añade una variable extra al paso de la función, con este label se arregla
    def scheduler(epoch):
        if epoch < 80:
            return 0.1
        if epoch < 160:
            return 0.01
        return 0.001

    def fit(self, parameters, config):
        self.actual_round += 1
        self.model.set_weights(parameters)
        change_lr = LearningRateScheduler(self.scheduler)
        H = self.model.fit(
            self.images["train"],
            self.labels["train"],
            epochs=self.epochs,
            validation_data=(self.images["valid"],self.labels["valid"]),
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[TensorBoard(log_dir=f"{PROJECT_PATH}/logs/fit/client_{self.partition_id}/round_{self.actual_round}"), change_lr]
        )

        # Get Train Metrics
        train_acc = H.history["accuracy"][-1]
        val_acc = H.history["val_accuracy"][-1]

        return self.model.get_weights(), len(self.images["train"]), {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.images["test"], self.labels["test"], verbose=1)
        return loss, len(self.images["test"]), {"test_accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    #data = load_data(partition_id, num_partitions)
    data = load_data(0, 1) # No partitions
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose, partition_id
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
