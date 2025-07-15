"""Flwr-Cifar10-basic: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import tensorflow as tf
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
        self.partition_id = partition_id

    @staticmethod
    def scheduler(epoch):
        eta_max = 0.01  # valor inicial
        eta_min = 0.001  # valor mínimo
        T_max = 310  # número total de epochs

        cosine_decay = 0.5 * (1 + tf.math.cos(tf.constant(epoch / T_max * 3.1415926535)))
        lr = eta_min + (eta_max - eta_min) * cosine_decay
        return float(lr)

    def fit(self, parameters, config):
        actual_round = config["round"]
        self.model.set_weights(parameters)
        change_lr = LearningRateScheduler(self.scheduler)
        H = self.model.fit(
            self.images["train"],
            self.labels["train"],
            epochs=self.epochs,
            validation_data=(self.images["valid"],self.labels["valid"]),
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[TensorBoard(log_dir=f"{PROJECT_PATH}/logs/fit/client_{self.partition_id}/round_{actual_round}"), change_lr]
        )

        # Get Train Metrics
        train_acc = H.history["accuracy"][-1]
        val_acc = H.history["val_accuracy"][-1]

        return self.model.get_weights(), len(self.images["train"]), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        # loss_train, acc_train = self.model.evaluate(self.images["train"], self.labels["train"], verbose=0)
        # loss_val, acc_val = self.model.evaluate(self.images["valid"], self.labels["valid"], verbose=0)
        loss_test, acc_test = self.model.evaluate(self.images["test"], self.labels["test"], verbose=0)

        return loss_test, len(self.images["test"]), {
            "test_accuracy": acc_test,
        }


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
