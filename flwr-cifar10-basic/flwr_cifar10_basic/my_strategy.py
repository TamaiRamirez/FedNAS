from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import json
from datetime import datetime

from .task import load_model, PROJECT_PATH
import tensorflow as tf  # Asegúrate de importar TensorFlow

class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases, and logging to TensorBoard.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dictionary to store metrics
        self.results_to_save = {}

        # TensorBoard summary writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{PROJECT_PATH}/logs/tensorboard/{current_time}"
        self.tb_writer = tf.summary.create_file_writer(log_dir)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate received model updates and save global model checkpoint."""
        
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Instantiate and load model
        model = load_model()
        model.set_weights(parameters_aggregated)
        model.save_weights(f"{PROJECT_PATH}/checkpoints/global_model_round_{server_round}")

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model and log metrics to JSON, W&B, and TensorBoard."""
        
        result = super().evaluate(server_round, parameters)

        if result is None:
            print(f"[Round {server_round}] No evaluation data available.")
            return None

        loss, metrics = result
        my_results = {"loss": loss, **metrics}
        self.results_to_save[server_round] = my_results

        # Save metrics to JSON
        with open("results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log to TensorBoard
        with self.tb_writer.as_default():
            for key, value in my_results.items():
                tf.summary.scalar(name=key, data=value, step=server_round)
            self.tb_writer.flush()

        # Mostrar en consola
        print(f"[Round {server_round}] Metrics:")
        for k, v in my_results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return loss, metrics

    def __del__(self):
        self.tb_writer.close()
