from flwr.common import FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from flwr.server.client_proxy import ClientProxy, EvaluateRes
from flwr.server.strategy import FedAvg, FedProx
from functools import partial, reduce

import json
from datetime import datetime
from datetime import timedelta
import shutil


from .task import PROJECT_PATH, NON_IID, ALPHA, DATASET_NAME, AGG_METHOD, ABLATION, BETA, NUM_NODES
import tensorflow as tf  # Asegúrate de importar TensorFlow
import os
import time
import numpy as np
import experiment_api as exp_api
import ablation_api as abla_api
import pandas as pd
import math

api = abla_api if ABLATION else exp_api

class CustomFedProxPerformace(FedProx):
    """A strategy that keeps the core functionality of FedProx with performance weightning and enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases, and logging to TensorBoard.
    """
    
    def __init__(self, num_rounds, initial_parameters=None, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)

        # Dictionary to store metrics
        self.results_to_save = {}
        self.start_time = time.time()
        self.num_rounds = num_rounds
        self.variation_metric = []  # Lista para almacenar la variación de la métrica entre rondas
        self.weights_entropy = []
        self.effective_num_clients = None  # número efectivo de clientes en cada ronda
        self.prev_f1_by_client: dict[str, float] = {}  # Historial de F1 scores para cada cliente en cada ronda

    def performance_aggregate(
        self,
        results: list[tuple[str, NDArrays, int, float]],
        server_round: int,
    ) -> NDArrays:
        """Compute performance weighted average."""

        round_variations = []
        current_f1_by_client = {}

        for client_id, _, _, f1_score in results:
            prev_f1 = self.prev_f1_by_client.get(client_id, 0.0)
            round_variations.append(f1_score - prev_f1)
            current_f1_by_client[client_id] = f1_score

        self.prev_f1_by_client = current_f1_by_client
        self.variation_metric.append(round_variations)

        weighted_results = []
        mean_var = np.mean(round_variations)
        std_var = np.std(round_variations)

        for idx, (client_id, weights, num_examples, _) in enumerate(results):
            z_score = np.clip(
                (round_variations[idx] - mean_var) / (std_var + 1e-8),
                -3,
                3,
            )
            lambda_weight = math.exp(BETA * z_score)
            weighted_results.append((client_id, weights, num_examples, lambda_weight))

        total = sum(num_examples * lambda_weight for _, _, num_examples, lambda_weight in weighted_results)

        self.effective_num_clients = 1.0 / sum(
            (num_examples * lambda_weight / total) ** 2
            for _, _, num_examples, lambda_weight in weighted_results
        )

        round_entropy = -sum(
            (num_examples * lambda_weight / total)
            * math.log(num_examples * lambda_weight / total + 1e-12)
            for _, _, num_examples, lambda_weight in weighted_results
        )
        self.weights_entropy.append(round_entropy)

        weighted_weights = [
            [layer * num_examples * lambda_weight for layer in weights]
            for _, weights, num_examples, lambda_weight in weighted_results
        ]

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / total
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
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        
        # Convert results
        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics["child_valid_f1_score"])
        #     for _, fit_res in results
        # ]
        
        weights_results = [
            (
                client_proxy.cid,
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
                float(next(iter(fit_res.metrics.values())))
            )
            for client_proxy, fit_res in results
        ]

        aggregated_ndarrays = self.performance_aggregate(weights_results, server_round)

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
        f1_scores_valid, f1_scores_test = [], []
        for _, res in results:
            weight = res.num_examples / total_examples
            loss_total += res.loss * weight

            for key, value in res.metrics.items():
                if key not in metrics_accum:
                    metrics_accum[key] = 0.0
                metrics_accum[key] += value * weight
                if key=="child_valid_f1_score":
                    f1_scores_valid.append(value)
                if key=="child_test_f1_score":
                    f1_scores_test.append(value)

        # Analisis de fairness
        CV_clients_valid = np.std(f1_scores_valid)/ (np.mean(f1_scores_valid) + 1e-12) # coeficiente de variación (CV) dispersion entre clientes
        Jain_index_valid = (np.sum(f1_scores_valid))**2 / (len(f1_scores_valid) * np.sum(np.array(f1_scores_valid)**2) + 1e-12)  # Índice de Jain (0 a 1, 1 es perfecto)
        CV_clients_test = np.std(f1_scores_test)/ (np.mean(f1_scores_test) + 1e-12) # coeficiente de variación (CV) dispersion entre clientes
        Jain_index_test = (np.sum(f1_scores_test))**2 / (len(f1_scores_test) * np.sum(np.array(f1_scores_test)**2) + 1e-12)  # Índice de Jain (0 a 1, 1 es perfecto)
        Shannon_entropy = self.weights_entropy[-1] / math.log(len(results)) if len(results) > 1 else 0.0  # Entropía de Shannon de la distribución de métricas entre clientes
        # A menor CV y mayor índice de Jain, mayor fairness.
        
        
        # Guardar resultados para esta ronda
        global_results = {
            "exp_id": api.get_exp_id(),
            "round": server_round,
            "loss": loss_total, 
            "CV_valid": CV_clients_valid, 
            "Jain_index_valid": Jain_index_valid,
            "CV_test": CV_clients_test, 
            "Jain_index_test": Jain_index_test,
            "Shannon_entropy": Shannon_entropy,
            "Effective_num_clients": round(self.effective_num_clients),
            **metrics_accum}

        path = api.get_results_dir() / f"global_metrics.parquet"
        # table = pa.Table.from_pylist([global_results])
        # if path.exists():
        #     pq.write_table(table, path, append=True)
        # else:
        #     pq.write_table(table, path)
        
        df = pd.DataFrame([global_results])
        
        if path.exists():
            df_existing = pd.read_parquet(path)
            df = pd.concat([df_existing, df], ignore_index=True)

        df.to_parquet(path, engine="pyarrow", index=False)

        # Mostrar en consola
        print(f"[Round {server_round}] Aggregated Metrics:")
        for k, v in global_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if server_round == self.num_rounds:  # última ronda
            exp_dir = api.get_experiment_dir()
            path = exp_dir / "metadata_experiment.txt"
            elapsed = time.time() - self.start_time
            elapsed_td = timedelta(seconds=int(elapsed))
            with open(path, "a") as f:
                f.write(f"Total training time: {elapsed_td} \n")
            print(f"⏱ Total training time: {elapsed_td}")

        return loss_total, metrics_accum
    
class CustomFedProx(FedProx):
    """A strategy that keeps the core functionality of FedProx unchanged but enables
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
        f1_scores_valid, f1_scores_test = [], []
        for _, res in results:
            weight = res.num_examples / total_examples
            loss_total += res.loss * weight

            for key, value in res.metrics.items():
                if key not in metrics_accum:
                    metrics_accum[key] = 0.0
                metrics_accum[key] += value * weight
                if key=="child_valid_f1_score":
                    f1_scores_valid.append(value)
                if key=="child_test_f1_score":
                    f1_scores_test.append(value)

        # Analisis de fairness
        CV_clients_valid = np.std(f1_scores_valid)/ (np.mean(f1_scores_valid) + 1e-12) # coeficiente de variación (CV) dispersion entre clientes
        Jain_index_valid = (np.sum(f1_scores_valid))**2 / (len(f1_scores_valid) * np.sum(np.array(f1_scores_valid)**2) + 1e-12)  # Índice de Jain (0 a 1, 1 es perfecto)
        CV_clients_test = np.std(f1_scores_test)/ (np.mean(f1_scores_test) + 1e-12) # coeficiente de variación (CV) dispersion entre clientes
        Jain_index_test = (np.sum(f1_scores_test))**2 / (len(f1_scores_test) * np.sum(np.array(f1_scores_test)**2) + 1e-12)  # Índice de Jain (0 a 1, 1 es perfecto)
        # A menor CV y mayor índice de Jain, mayor fairness.
        
        
        # Guardar resultados para esta ronda
        global_results = {
            "exp_id": api.get_exp_id(),
            "round": server_round,
            "loss": loss_total, 
            "CV_valid": CV_clients_valid, 
            "Jain_index_valid": Jain_index_valid,
            "CV_test": CV_clients_test, 
            "Jain_index_test": Jain_index_test,
            **metrics_accum}

        path = api.get_results_dir() / f"global_metrics.parquet"
        # table = pa.Table.from_pylist([global_results])
        # if path.exists():
        #     pq.write_table(table, path, append=True)
        # else:
        #     pq.write_table(table, path)
        
        df = pd.DataFrame([global_results])
        
        if path.exists():
            df_existing = pd.read_parquet(path)
            df = pd.concat([df_existing, df], ignore_index=True)

        df.to_parquet(path, engine="pyarrow", index=False)

        # Mostrar en consola
        print(f"[Round {server_round}] Aggregated Metrics:")
        for k, v in global_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if server_round == self.num_rounds:  # última ronda
            exp_dir = api.get_experiment_dir()
            path = exp_dir / "metadata_experiment.txt"
            elapsed = time.time() - self.start_time
            elapsed_td = timedelta(seconds=int(elapsed))
            with open(path, "a") as f:
                f.write(f"Total training time: {elapsed_td} \n")
            print(f"⏱ Total training time: {elapsed_td}")

        return loss_total, metrics_accum
    
class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases, and logging to TensorBoard.
    """

    def __init__(self, num_rounds, initial_parameters=None, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)

        # Dictionary to store metrics
        # self.results_to_save = {}
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
        f1_scores_valid, f1_scores_test = [], []
        for _, res in results:
            weight = res.num_examples / total_examples
            loss_total += res.loss * weight

            for key, value in res.metrics.items():
                if key not in metrics_accum:
                    metrics_accum[key] = 0.0
                metrics_accum[key] += value * weight
                if key=="child_valid_f1_score":
                    f1_scores_valid.append(value)
                if key=="child_test_f1_score":
                    f1_scores_test.append(value)

        # Analisis de fairness
        CV_clients_valid = np.std(f1_scores_valid)/ (np.mean(f1_scores_valid) + 1e-12) # coeficiente de variación (CV) dispersion entre clientes
        Jain_index_valid = (np.sum(f1_scores_valid))**2 / (len(f1_scores_valid) * np.sum(np.array(f1_scores_valid)**2) + 1e-12)  # Índice de Jain (0 a 1, 1 es perfecto)
        CV_clients_test = np.std(f1_scores_test)/ (np.mean(f1_scores_test) + 1e-12) # coeficiente de variación (CV) dispersion entre clientes
        Jain_index_test = (np.sum(f1_scores_test))**2 / (len(f1_scores_test) * np.sum(np.array(f1_scores_test)**2) + 1e-12)  # Índice de Jain (0 a 1, 1 es perfecto)
        # A menor CV y mayor índice de Jain, mayor fairness.
        
        
        # Guardar resultados para esta ronda
        global_results = {
            "exp_id": api.get_exp_id(),
            "round": server_round,
            "loss": loss_total, 
            "CV_valid": CV_clients_valid, 
            "Jain_index_valid": Jain_index_valid,
            "CV_test": CV_clients_test, 
            "Jain_index_test": Jain_index_test,
            **metrics_accum}

        path = api.get_results_dir() / f"global_metrics.parquet"
        # table = pa.Table.from_pylist([global_results])
        # if path.exists():
        #     pq.write_table(table, path, append=True)
        # else:
        #     pq.write_table(table, path)
        
        df = pd.DataFrame([global_results])
        
        if path.exists():
            df_existing = pd.read_parquet(path)
            df = pd.concat([df_existing, df], ignore_index=True)

        df.to_parquet(path, engine="pyarrow", index=False)

        # Mostrar en consola
        print(f"[Round {server_round}] Aggregated Metrics:")
        for k, v in global_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if server_round == self.num_rounds:  # última ronda
            exp_dir = api.get_experiment_dir()
            path = exp_dir / "metadata_experiment.txt"
            elapsed = time.time() - self.start_time
            elapsed_td = timedelta(seconds=int(elapsed))
            with open(path, "a") as f:
                f.write(f"Total training time: {elapsed_td} \n")
            print(f"⏱ Total training time: {elapsed_td}")

        return loss_total, metrics_accum