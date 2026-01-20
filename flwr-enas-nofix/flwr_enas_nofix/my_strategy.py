from flwr.common import FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from flwr.server.client_proxy import ClientProxy, EvaluateRes
from flwr.server.strategy import FedAvg
from functools import partial, reduce

import json
from datetime import datetime
from datetime import timedelta
import shutil


from .task import PROJECT_PATH, NON_IID, ALPHA, DATASET_NAME, AGG_METHOD
import tensorflow as tf  # Asegúrate de importar TensorFlow
import os
import time
import numpy as np

class CustomFedProxPerformace(FedAvg):
    """A strategy that keeps the core functionality of FedAvg with performance weightning and enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases, and logging to TensorBoard.
    """

    def __init__(self, num_rounds, initial_parameters=None, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)

        # Dictionary to store metrics
        self.results_to_save = {}
        self.start_time = time.time()
        self.num_rounds = num_rounds
        
    def performance_aggregate(self, results: list[tuple[NDArrays, int]], total_acc) -> NDArrays:
        """Compute performance weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for (_, num_examples, _) in results)

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples * acc/total_acc for layer in weights] for weights, num_examples, acc in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights, strict=True)
        ]
        return weights_prime

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate received model updates and save global model checkpoint."""
        """ This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""
        # parameters_aggregated, metrics_aggregated = super().aggregate_fit(
        #     server_round, results, failures
        # )
        
        test_acc = []
        for _, res in results:
            test_acc.append(float(next(iter(res.metrics.values()))))
        total_f1 = max(test_acc)
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, float(next(iter(res.metrics.values()))))
            for _, fit_res in results
        ]
        aggregated_ndarrays = self.performance_aggregate(weights_results, total_f1)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        # if self.fit_metrics_aggregation_fn:
        #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)


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
        test_f1 = []
        for _, res in results:
            weight = res.num_examples / total_examples
            loss_total += res.loss * weight

            for key, value in res.metrics.items():
                if key not in metrics_accum:
                    metrics_accum[key] = 0.0
                metrics_accum[key] += value * weight
                if key=="child_test_f1_score":
                    test_f1.append(value * weight)

        CV = np.std(test_f1)/np.mean(test_f1) # coeficiente de variación (CV) F1 Score

        # Guardar resultados para esta ronda
        round_results = {"loss": loss_total, **metrics_accum, "CV": CV}
        self.results_to_save[server_round] = round_results
        # Directorio de guardado
        if NON_IID:
            save_path=f"{PROJECT_PATH}/{AGG_METHOD}/{DATASET_NAME}/Non-IID/{ALPHA}"
        else:
            save_path=f"{PROJECT_PATH}/{AGG_METHOD}/{DATASET_NAME}/IID"

        # Guardar en JSON
        with open(f"{save_path}/results_global.json", "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        # Mostrar en consola
        print(f"[Round {server_round}] Aggregated Metrics:")
        for k, v in round_results.items():
            print(f"  {k}: {v:.4f}")

        if server_round == self.num_rounds:  # última ronda
            elapsed = time.time() - self.start_time
            elapsed_td = timedelta(seconds=int(elapsed))
            with open(f"{save_path}/Training_time.txt", 'w', encoding='utf-8') as archivo:
                archivo.write(f"Tiempo Total de entrenamiento: {elapsed_td}")
            print(f"⏱ Tiempo total de entrenamiento: {elapsed_td}")

        return loss_total, metrics_accum
    
class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases, and logging to TensorBoard.
    """
    # PENDIENTE DE IMPLEMENTAR
    None
    
class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases, and logging to TensorBoard.
    """

    def __init__(self, num_rounds, initial_parameters=None, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)

        # Dictionary to store metrics
        self.results_to_save = {}
        self.start_time = time.time()
        self.num_rounds = num_rounds

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate received model updates and save global model checkpoint."""
        """ This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
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
        test_accuracies = []
        for _, res in results:
            weight = res.num_examples / total_examples
            loss_total += res.loss * weight

            for key, value in res.metrics.items():
                if key not in metrics_accum:
                    metrics_accum[key] = 0.0
                metrics_accum[key] += value * weight
                if key=="child_test_acc":
                    test_accuracies.append(value * weight)

        CV = np.std(test_accuracies)/np.mean(test_accuracies) # coeficiente de variación (CV)

        # Guardar resultados para esta ronda
        round_results = {"loss": loss_total, **metrics_accum, "CV": CV}
        self.results_to_save[server_round] = round_results
        # Directorio de guardado
        if NON_IID:
            save_path=f"{PROJECT_PATH}/{AGG_METHOD}/{DATASET_NAME}/Non-IID/{ALPHA}"
        else:
            save_path=f"{PROJECT_PATH}/{AGG_METHOD}/{DATASET_NAME}/IID"

        # Guardar en JSON
        with open(f"{save_path}/results_global.json", "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        # Mostrar en consola
        print(f"[Round {server_round}] Aggregated Metrics:")
        for k, v in round_results.items():
            print(f"  {k}: {v:.4f}")

        if server_round == self.num_rounds:  # última ronda
            elapsed = time.time() - self.start_time
            elapsed_td = timedelta(seconds=int(elapsed))
            with open(f"{save_path}/Training_time.txt", 'w', encoding='utf-8') as archivo:
                archivo.write(f"Tiempo Total de entrenamiento: {elapsed_td}")
            print(f"⏱ Tiempo total de entrenamiento: {elapsed_td}")

        return loss_total, metrics_accum