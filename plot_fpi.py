#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


EPSILON = 1e-12

# Pesos del FPI
ALPHA = 0.1
BETA = 0.25
GAMMA = 0.65

SENSITIVITY_WEIGHTS = [
    (0.10, 0.25, 0.65),
    (0.10, 0.30, 0.60),
    (0.15, 0.25, 0.60),
    (0.05, 0.30, 0.65),
    (0.10, 0.20, 0.70),
    (0.20, 0.30, 0.50),
]

VALID_METHODS = {"fedavg", "fedprox", "custom"}

METHOD_DISPLAY = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "custom": "FedFPA",
}

METHOD_COLORS = {
    "fedavg": "#1f77b4",
    "fedprox": "#d62728",
    "custom": "#2ca02c",
}

SCENARIO_ORDER = ["IID", "Non_IID_1.0", "Non_IID_0.5", "Non_IID_0.1"]


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (13, 5),
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "font.size": 20,
        "axes.titlesize": 12,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def parse_metadata_file(metadata_path: Path) -> Optional[Dict[str, str]]:
    if not metadata_path.exists():
        return None

    text = metadata_path.read_text(encoding="utf-8", errors="ignore")
    data = {}

    patterns = {
        "experiment_id": r"EXPERIMENT ID:\s*(.+)",
        "status": r"STATUS:\s*(.+)",
        "dataset_name": r"DATASET_NAME:\s*(.+)",
        "data_partitioning": r"DATA PARTITIONING:\s*(.+)",
        "agg_method": r"AGG_METHOD:\s*(.+)",
        "num_rounds": r"NUM_ROUNDS:\s*(.+)",
        "num_nodes": r"NUM_NODES:\s*(.+)",
        "mu": r"MU:\s*(.+)",
        "batch_size": r"BATCH_SIZE:\s*(.+)",
        "local_epochs": r"LOCAL_EPOCHS:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = match.group(1).strip()

    required = ["status", "dataset_name", "data_partitioning", "agg_method"]
    if any(k not in data for k in required):
        return None

    return data


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower()

"""This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""

def parse_scenario(data_partitioning: str) -> Optional[str]:
    value = data_partitioning.strip()

    if value.upper() == "IID":
        return "IID"

    match = re.search(r"Non-IID\s+with\s+alpha\s*=\s*([0-9]*\.?[0-9]+)", value, re.IGNORECASE)
    if match:
        alpha = float(match.group(1))
        return f"Non_IID_{alpha:.1f}"

    return None


def scenario_pretty_name(scenario: str) -> str:
    mapping = {
        "IID": "IID",
        "Non_IID_1.0": "Non-IID, Alpha = 1.0",
        "Non_IID_0.5": "Non-IID, Alpha = 0.5",
        "Non_IID_0.1": "Non-IID, Alpha = 0.1",
    }
    return mapping.get(scenario, scenario)


def is_experiment_dir(path: Path) -> bool:
    return path.is_dir() and re.fullmatch(r"\d{8}_\d+", path.name) is not None


def load_experiment_clients(results_dir: Path) -> Optional[pd.DataFrame]:
    parquet_files = sorted(results_dir.glob("client_*.parquet"))
    if not parquet_files:
        return None

    dfs = []
    for p in parquet_files:
        df = pd.read_parquet(p)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def discover_valid_experiments(root_dir: Path, dataset_name: str) -> Dict[str, Dict[str, Path]]:
    """
    Devuelve:
      {
        scenario: {
          method: experiment_path
        }
      }

    Si hay varios experimentos VALID para el mismo scenario+método,
    se selecciona el más reciente por nombre de carpeta.
    """
    dataset_norm = normalize_dataset_name(dataset_name)
    selected: Dict[str, Dict[str, Path]] = {}

    for child in root_dir.iterdir():
        if not is_experiment_dir(child):
            continue

        metadata_path = child / "metadata_experiment.txt"
        results_dir = child / "results"

        if not metadata_path.exists() or not results_dir.exists():
            continue

        metadata = parse_metadata_file(metadata_path)
        if metadata is None:
            continue

        if metadata["status"].strip().upper() != "VALID":
            continue

        if normalize_dataset_name(metadata["dataset_name"]) != dataset_norm:
            continue

        method = metadata["agg_method"].strip().lower()
        if method not in VALID_METHODS:
            continue

        scenario = parse_scenario(metadata["data_partitioning"])
        if scenario not in SCENARIO_ORDER:
            continue

        selected.setdefault(scenario, {})
        previous = selected[scenario].get(method)
        if previous is None or child.name > previous.name:
            selected[scenario][method] = child

    return selected


def compute_round_fpi_stats(
    clients_df: pd.DataFrame,
    score_col: str,
    loss_col: str,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> pd.DataFrame:
    """
    Calcula por ronda:
      mu_f    = media del score entre clientes
      sigma_f = desviación típica poblacional del score entre clientes
      CV_f    = sigma_f / (mu_f + eps)
      Loss_f  = media de la loss entre clientes
      FPI     = (mu_f^alpha) * (1/(1+Loss_f))^beta * (1/(1+CV_f))^gamma

    También guarda el peor cliente por ronda para el score.
    """
    required_cols = {"round", "client_id", score_col, loss_col}
    missing = required_cols - set(clients_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas {missing} en los parquet de clientes.")

    rows = []

    for round_id, g in clients_df.groupby("round"):
        scores = g[score_col].astype(float)
        losses = g[loss_col].astype(float)

        mu_f = scores.mean()
        sigma_f = math.sqrt(((scores - mu_f) ** 2).mean())
        cv_f = sigma_f / (abs(mu_f) + EPSILON)
        loss_f = losses.mean()

        # Para evitar problemas si mu_f fuera negativo por ruido, lo acotamos
        mu_component = max(mu_f, 0.0)

        fpi = (
            (mu_component ** alpha)
            * ((1.0 / (1.0 + loss_f)) ** beta)
            * ((1.0 / (1.0 + cv_f)) ** gamma)
        )

        worst_idx = scores.idxmin()
        worst_client_id = g.loc[worst_idx, "client_id"]
        worst_client_score = g.loc[worst_idx, score_col]

        rows.append({
            "round": int(round_id),
            "mu_f": float(mu_f),
            "sigma_f": float(sigma_f),
            "cv_f": float(cv_f),
            "loss_f": float(loss_f),
            "fpi": float(fpi),
            "worst_client_id": int(worst_client_id) if pd.notna(worst_client_id) else None,
            "worst_client_score": float(worst_client_score),
        })

    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def plot_scenario_fpi_metric(
    scenario: str,
    scenario_results: Dict[str, Dict[str, pd.DataFrame]],
    metric_key: str,
    output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in ["fedavg", "fedprox", "custom"]:
        df = scenario_results.get(method, {}).get(metric_key)
        if df is None or df.empty:
            continue

        ax.plot(
            df["round"],
            df["fpi"],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=METHOD_COLORS[method],
            label=METHOD_DISPLAY[method]
        )

    # ax.set_title(metric_title, pad=8)
    ax.set_xlabel("Round", fontsize=20)
    ax.set_ylabel("FPI", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="best",
            ncol=1,
            fontsize=20,
            frameon=True,
            handlelength=2.2,
            borderaxespad=0.8,
            borderpad=0.6,
            labelspacing=0.5,
        )

    # fig.suptitle(f"Fairness-Performance Index — {scenario_pretty_name(scenario)}", y=1.08, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_weight_for_filename(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def plot_sensitivity_test_fpi(
    scenario_results_all: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    weights: Tuple[float, float, float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    scenario_results = scenario_results_all.get("Non_IID_0.5", {})

    for method in ["fedavg", "fedprox", "custom"]:
        df = scenario_results.get(method, {}).get("test_f1_score")
        if df is None or df.empty:
            continue

        ax.plot(
            df["round"],
            df["fpi"],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=METHOD_COLORS[method],
            label=METHOD_DISPLAY[method],
        )

    ax.set_xlabel("Round", fontsize=20)
    ax.set_ylabel("FPI", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend(
            handles,
            labels,
            loc="best",
            ncol=1,
            fontsize=20,
            frameon=True,
            handlelength=2.2,
            borderaxespad=0.8,
            borderpad=0.6,
            labelspacing=0.5,
        )

    alpha, beta, gamma = weights
    # fig.suptitle(f"Test FPI sensitivity: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}", fontsize=20)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_last_round_summary(
    scenario_results_all: Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
) -> pd.DataFrame:
    rows = []

    for scenario in SCENARIO_ORDER:
        methods_map = scenario_results_all.get(scenario, {})

        for method in ["fedavg", "fedprox", "custom"]:
            val_df = methods_map.get(method, {}).get("val_f1_score")
            test_df = methods_map.get(method, {}).get("test_f1_score")

            row = {
                "scenario": scenario,
                "method": METHOD_DISPLAY[method],
                "last_round": None,
                "val_fpi_last_round": None,
                "test_fpi_last_round": None,
                "val_worst_client_id_last_round": None,
                "val_worst_client_f1_last_round": None,
                "test_worst_client_id_last_round": None,
                "test_worst_client_f1_last_round": None,
            }

            last_rounds = []

            if val_df is not None and not val_df.empty:
                last_val = val_df.sort_values("round").iloc[-1]
                row["val_fpi_last_round"] = float(last_val["fpi"])
                row["val_worst_client_id_last_round"] = int(last_val["worst_client_id"]) if pd.notna(last_val["worst_client_id"]) else None
                row["val_worst_client_f1_last_round"] = float(last_val["worst_client_score"])
                last_rounds.append(int(last_val["round"]))

            if test_df is not None and not test_df.empty:
                last_test = test_df.sort_values("round").iloc[-1]
                row["test_fpi_last_round"] = float(last_test["fpi"])
                row["test_worst_client_id_last_round"] = int(last_test["worst_client_id"]) if pd.notna(last_test["worst_client_id"]) else None
                row["test_worst_client_f1_last_round"] = float(last_test["worst_client_score"])
                last_rounds.append(int(last_test["round"]))

            if last_rounds:
                row["last_round"] = max(last_rounds)

            rows.append(row)

    return pd.DataFrame(rows)


def build_scenario_results(
    experiments: Dict[str, Dict[str, Path]],
    include_validation: bool = True,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    scenario_results_all: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    for scenario in SCENARIO_ORDER:
        if scenario not in experiments:
            continue

        scenario_results_all.setdefault(scenario, {})

        for method, exp_path in experiments[scenario].items():
            results_dir = exp_path / "results"
            clients_df = load_experiment_clients(results_dir)
            if clients_df is None or clients_df.empty:
                continue

            test_df = compute_round_fpi_stats(
                clients_df=clients_df,
                score_col="test_f1_score",
                loss_col="test_loss",
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            scenario_results_all[scenario][method] = {
                "test_f1_score": test_df,
            }

            if include_validation:
                val_df = compute_round_fpi_stats(
                    clients_df=clients_df,
                    score_col="val_f1_score",
                    loss_col="val_loss",
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                scenario_results_all[scenario][method]["val_f1_score"] = val_df

    return scenario_results_all


def run_sensitivity_analysis(
    experiments: Dict[str, Dict[str, Path]],
    output_dir: Path,
) -> None:
    sensitivity_dir = output_dir / "Sensitivity_analisis"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_experiments = {
        "Non_IID_0.5": experiments.get("Non_IID_0.5", {})
    }

    if not sensitivity_experiments["Non_IID_0.5"]:
        print("[INFO] No se encontraron experimentos para Non-IID with alpha=0.5.")
        return

    for weights in SENSITIVITY_WEIGHTS:
        alpha, beta, gamma = weights
        scenario_results_all = build_scenario_results(
            sensitivity_experiments,
            include_validation=False,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        alpha_str = format_weight_for_filename(alpha)
        beta_str = format_weight_for_filename(beta)
        gamma_str = format_weight_for_filename(gamma)
        plot_path = sensitivity_dir / f"fpi_{alpha_str}_{beta_str}_{gamma_str}.png"
        plot_sensitivity_test_fpi(scenario_results_all, weights, plot_path)
        print(f"[OK] Guardado sensitivity plot: {plot_path}")


def main(root_dir: str, dataset_name: str, sensitivity: bool = False) -> None:
    set_plot_style()

    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"El directorio raíz no existe: {root}")

    output_dir = root.parent / "Plots" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_valid_experiments(root, dataset_name)

    if not experiments:
        print(f"No se encontraron experimentos VALID para el dataset '{dataset_name}'.")
        return

    if sensitivity:
        run_sensitivity_analysis(experiments, output_dir)
        return

    scenario_results_all = build_scenario_results(experiments)

    for scenario in SCENARIO_ORDER:
        if scenario not in scenario_results_all:
            continue

        if scenario_results_all[scenario]:
            plots_to_generate = {
                "validation": "val_f1_score",
                "test": "test_f1_score",
            }

            for split_name, metric_key in plots_to_generate.items():
                plot_path = output_dir / f"{scenario}_fpi_{split_name}.png"
                plot_scenario_fpi_metric(
                    scenario,
                    scenario_results_all[scenario],
                    metric_key,
                    plot_path,
                )
                print(f"[OK] Guardado plot: {plot_path}")

    summary_df = build_last_round_summary(scenario_results_all)
    csv_path = output_dir / f"{dataset_name}_fpi_last_round.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[OK] Guardado CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcula y plotea el FPI para experimentos de Federated Learning."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Directorio raíz que contiene los subdirectorios YYYYMMDD_X"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help='Nombre del dataset, por ejemplo: "cifar10"'
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Activa el sensitivity analysis variando alpha, beta y gamma solo para test."
    )

    args = parser.parse_args()
    main(args.root_dir, args.dataset, args.sensitivity)
