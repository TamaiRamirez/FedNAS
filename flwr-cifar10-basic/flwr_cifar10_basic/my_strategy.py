from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy, EvaluateRes
from flwr.server.strategy import FedAvg
import json
from datetime import datetime
from datetime import timedelta
import shutil

from .task import load_model, PROJECT_PATH, MODEL
import tensorflow as tf  # Asegúrate de importar TensorFlow
import os
import time

class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases, and logging to TensorBoard.
    """

    def __init__(self, num_rounds, model='custom', *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dictionary to store metrics
        self.results_to_save = {}
        self.start_time = time.time()
        self.num_rounds = num_rounds
        self.model = model
        # TensorBoard summary writer
        if os.path.isdir(f"{PROJECT_PATH}/logs/{MODEL}/tensorboard/"):
            shutil.rmtree(f"{PROJECT_PATH}/logs/{MODEL}/tensorboard/")
        if os.path.isdir(f"{PROJECT_PATH}/logs/{MODEL}/fit/"):
            shutil.rmtree(f"{PROJECT_PATH}/logs/{MODEL}/fit/")
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{PROJECT_PATH}/logs/{MODEL}/tensorboard/{current_time}"
        self.tb_writer = tf.summary.create_file_writer(log_dir)

    # def configure_fit(self, server_round, parameters, client_manager):
    #     if self.start_time is None:
    #         self.start_time = time.time()  # marca el inicio justo antes de la primera ronda
    #     return super().configure_fit(server_round, parameters, client_manager)
    """ This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""
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
            
        # # Instantiate and load model
        # model = load_model()
        # weights = parameters_to_ndarrays(parameters_aggregated)
        # model.set_weights(weights)
        # model.save_weights(f"{PROJECT_PATH}/checkpoints/global_model_round_{server_round}.h5")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[BaseException],
    ) -> tuple[float, dict[str, float]]:

        if not results:
            print(f"[Round {server_round}] No evaluation results received.")
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        
        # Inicializar acumuladores
        loss_total = 0.0
        metrics_accum = {}

        for _, res in results:
            weight = res.num_examples / total_examples
            loss_total += res.loss * weight

            for key, value in res.metrics.items():
                if key not in metrics_accum:
                    metrics_accum[key] = 0.0
                metrics_accum[key] += value * weight

        # Guardar resultados para esta ronda
        round_results = {"loss": loss_total, **metrics_accum}
        self.results_to_save[server_round] = round_results

        # Guardar en JSON
        with open(f"{PROJECT_PATH}/results_{MODEL}.json", "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        # TensorBoard
        with self.tb_writer.as_default():
            for key, value in round_results.items():
                tf.summary.scalar(name=key, data=value, step=server_round)
            self.tb_writer.flush()

        # Mostrar en consola
        print(f"[Round {server_round}] Aggregated Metrics:")
        for k, v in round_results.items():
            print(f"  {k}: {v:.4f}")

        if server_round == self.num_rounds:  # última ronda
            elapsed = time.time() - self.start_time
            elapsed_td = timedelta(seconds=int(elapsed))
            with open(f"{PROJECT_PATH}/Training_time_{MODEL}.txt", 'w', encoding='utf-8') as archivo:
                archivo.write(f"Tiempo Total de entrenamiento: {elapsed_td}")
            print(f"⏱ Tiempo total de entrenamiento: {elapsed_td}")

        

        return loss_total, metrics_accum

    def __del__(self):
        self.tb_writer.close()
