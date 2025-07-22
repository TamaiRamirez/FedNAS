"""Flwr-Cifar10-basic: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, Callback
from flwr_cifar10_basic.task import load_data, load_model, PROJECT_PATH, MODEL
import matplotlib.pyplot as plt
import os

class CosineAnnealingWithRestarts:
    def __init__(self, eta_max=0.05, eta_min=0.0005, T_0=10, T_mul=2):
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_mul = T_mul
        self.T_i = T_0
        self.epoch_since_restart = 0

    def __call__(self, epoch):
        if epoch != 0 and epoch % self.T_i == 0:
            self.epoch_since_restart = 0
            self.T_i *= self.T_mul  # Incrementar periodo
        else:
            self.epoch_since_restart += 1

        cosine_decay = 0.5 * (1 + tf.math.cos(
            tf.constant(self.epoch_since_restart / self.T_i * 3.1415926535)
        ))
        lr = self.eta_min + (self.eta_max - self.eta_min) * cosine_decay
        return float(lr)

class LRSchedulerLogger(Callback):
    def __init__(self):
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        if hasattr(self.model.optimizer, "learning_rate"):
            lr = float(self.model.optimizer.learning_rate)
        self.lrs.append(lr)

def smooth_curve(points, factor=0.8):
    """Aplica una media móvil exponencial a los puntos."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

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

    # @staticmethod
    # def scheduler(epoch):
    #     eta_max = 0.01  # valor inicial
    #     eta_min = 0.0001  # valor mínimo
    #     T_max = 310  # número total de epochs

    #     cosine_decay = 0.5 * (1 + tf.math.cos(tf.constant(epoch / T_max * 3.1415926535)))
    #     lr = eta_min + (eta_max - eta_min) * cosine_decay
    #     return float(lr)

    def fit(self, parameters, config):
        actual_round = config["round"]
        self.model.set_weights(parameters)
        scheduler = CosineAnnealingWithRestarts(
            eta_max=0.05,
            eta_min=0.0005,
            T_0=63,
            T_mul=1
        )
        lr_logger = LRSchedulerLogger()
        change_lr = LearningRateScheduler(scheduler)
        H = self.model.fit(
            self.images["train"],
            self.labels["train"],
            epochs=self.epochs,
            validation_data=(self.images["valid"],self.labels["valid"]),
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[TensorBoard(log_dir=f"{PROJECT_PATH}/logs/{MODEL}/fit/client_{self.partition_id}/round_{actual_round}"), change_lr, lr_logger]
        )

        # # Get Train Metrics
        # train_acc = H.history["accuracy"][-1]
        # val_acc = H.history["val_accuracy"][-1]

        # Directorio de guardado
        plot_dir = f"{PROJECT_PATH}/plots/{MODEL}/client_{self.partition_id}/round_{actual_round}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Gráfico de precisión (accuracy)
        plt.figure()
        plt.plot(smooth_curve(H.history["accuracy"], factor=0.5), label="Train Accuracy")
        plt.plot(smooth_curve(H.history["val_accuracy"], factor=0.55), label="Validation Accuracy")
        plt.title(f"Client {self.partition_id} - Accuracy (Round {actual_round})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/accuracy_round_{actual_round}.png")
        plt.close()

        # Gráfico de pérdida (loss)
        plt.figure()
        plt.plot(smooth_curve(H.history["loss"], factor=0.5), label="Train Loss")
        plt.plot(smooth_curve(H.history["val_loss"], factor=0.55), label="Validation Loss")
        plt.title(f"Client {self.partition_id} - Loss (Round {actual_round})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/loss_round_{actual_round}.png")
        plt.close()

        # Grafico Learning Rate
        plt.figure()
        plt.plot(lr_logger.lrs, label="Learning Rate")
        plt.title(f"Client {self.partition_id} - Learning Rate (Round {actual_round})")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/lr_round_{actual_round}.png")
        plt.close()

        return self.model.get_weights(), len(self.images["train"]), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        print("Evaluating Model Performance...")
        # loss_train, acc_train = self.model.evaluate(self.images["train"], self.labels["train"], verbose=0)
        loss_val, acc_val = self.model.evaluate(self.images["valid"], self.labels["valid"], verbose=1)
        loss_test, acc_test = self.model.evaluate(self.images["test"], self.labels["test"], verbose=1)

        return loss_test, len(self.images["test"]), {
            # "loss_train": loss_train,
            "loss_val": loss_val,
            "loss_test": loss_test,
            # "train_accuracy": acc_train,
            "val_accuracy": acc_val,
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
