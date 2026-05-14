from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIGURACIÓN PRINCIPAL
# =========================

# Directorio raíz donde están las carpetas YYYYMMDD_X
EXPERIMENTS_ROOT = Path("/workspace/flwr-enas-nofix/experiments")

# Directorio raíz de salida
PLOTS_ROOT = Path("/workspace/flwr-enas-nofix/Plots")

# Métodos permitidos y nombre a mostrar en leyenda
VALID_AGG_METHODS = {"fedavg", "fedprox", "custom"}
DISPLAY_NAMES = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "custom": "FedFPA",
}

# Escenarios permitidos y nombre de archivo
VALID_SCENARIOS = {
    "IID": "IID",
    "Non_IID_1.0": "Non_IID_1.0",
    "Non_IID_0.5": "Non_IID_0.5",
    "Non_IID_0.1": "Non_IID_0.1",
}

# Métricas a excluir
EXCLUDED_METRICS = {"Shannon_entropy", "Effective_num_clients", "loss"}

# Columnas base que no se consideran métricas a plotear
BASE_COLUMNS = {"exp_id", "round"}


# =========================
# PARSEO DE METADATOS
# =========================

def parse_metadata_file(metadata_path: Path) -> Optional[Dict[str, str]]:
    """
    Parseo robusto del archivo metadata_experiment.txt.

    Devuelve un diccionario con claves:
    - EXPERIMENT ID
    - STATUS
    - REASON
    - DATE
    - DATASET_NAME
    - DATA PARTITIONING
    - AGG_METHOD
    - NUM_ROUNDS
    - NUM_NODES
    - MU (si existe)
    - BATCH_SIZE
    - LOCAL_EPOCHS
    """
    if not metadata_path.exists():
        return None

    info: Dict[str, str] = {}

    with metadata_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            # Quitar tabulaciones iniciales
            stripped = line.strip()

            # Ignorar cabecera
            if stripped == "CONFIGURATION:":
                continue

            if ":" in stripped:
                key, value = stripped.split(":", 1)
                info[key.strip()] = value.strip()

    return info


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower()


def parse_scenario(data_partitioning: str) -> Optional[str]:
    """
    Convierte el texto de DATA PARTITIONING en uno de:
    - IID
    - Non_IID_1.0
    - Non_IID_0.5
    - Non_IID_0.1
    """
    dp = data_partitioning.strip()

    if dp == "IID":
        return "IID"

    # Esperado: "Non-IID with alpha=1.0"
    match = re.search(r"Non-IID\s+with\s+alpha\s*=\s*([0-9]*\.?[0-9]+)", dp, re.IGNORECASE)
    if match:
        alpha = float(match.group(1))
        if alpha in {1.0, 0.5, 0.1}:
            return f"Non_IID_{alpha:.1f}"

    return None


def is_valid_experiment_dir(path: Path) -> bool:
    """
    Comprueba que el nombre del directorio sigue el patrón YYYYMMDD_X
    """
    return path.is_dir() and re.fullmatch(r"\d{8}_\d+", path.name) is not None


# =========================
# RECOLECCIÓN DE EXPERIMENTOS
# =========================

def collect_experiments(root_dir: Path, dataset_name: str) -> List[Dict]:
    """
    Recorre los experimentos y devuelve una lista con la información necesaria.
    """
    dataset_name_norm = normalize_dataset_name(dataset_name)
    collected = []

    for exp_dir in root_dir.iterdir():
        if not is_valid_experiment_dir(exp_dir):
            continue

        metadata_path = exp_dir / "metadata_experiment.txt"
        results_path = exp_dir / "results" / "global_metrics.parquet"

        if not metadata_path.exists() or not results_path.exists():
            continue

        metadata = parse_metadata_file(metadata_path)
        if metadata is None:
            continue

        status = metadata.get("STATUS", "")
        dataset = metadata.get("DATASET_NAME", "")
        agg_method = metadata.get("AGG_METHOD", "")
        data_partitioning = metadata.get("DATA PARTITIONING", "")

        if normalize_dataset_name(dataset) != dataset_name_norm:
            continue

        if status != "VALID":
            continue

        if agg_method not in VALID_AGG_METHODS:
            continue

        scenario = parse_scenario(data_partitioning)
        if scenario not in VALID_SCENARIOS:
            continue

        collected.append(
            {
                "exp_dir": exp_dir,
                "metadata": metadata,
                "results_path": results_path,
                "agg_method": agg_method,
                "scenario": scenario,
            }
        )

    return collected


# =========================
# CARGA DE MÉTRICAS
# =========================

def load_global_metrics(parquet_path: Path) -> Optional[pd.DataFrame]:
    """
    Carga el parquet de métricas globales.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[WARNING] No se pudo leer {parquet_path}: {e}")
        return None

    if "round" not in df.columns:
        print(f"[WARNING] El archivo {parquet_path} no contiene la columna 'round'.")
        return None

    return df


def get_metrics_to_plot(experiments: List[Dict]) -> List[str]:
    """
    Obtiene el conjunto de métricas a plotear a partir de todos los dataframes,
    excluyendo columnas base y métricas prohibidas.
    """
    metrics = set()

    for exp in experiments:
        df = load_global_metrics(exp["results_path"])
        if df is None:
            continue

        for col in df.columns:
            if col in BASE_COLUMNS:
                continue
            if col in EXCLUDED_METRICS:
                continue
            metrics.add(col)

    return sorted(metrics)


def combine_experiments_by_metric(experiments: List[Dict], metric: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Devuelve una estructura:
    {
        scenario: {
            agg_method: dataframe_con_columnas[round, mean, std]
        }
    }

    Para cada escenario y método:
    - concatena todos los experimentos disponibles
    - agrupa por round
    - calcula mean y std
    """
    combined: Dict[str, Dict[str, pd.DataFrame]] = {
        scenario: {} for scenario in VALID_SCENARIOS
    }

    for scenario in VALID_SCENARIOS:
        for agg_method in VALID_AGG_METHODS:
            frames = []

            relevant_exps = [
                exp for exp in experiments
                if exp["scenario"] == scenario and exp["agg_method"] == agg_method
            ]

            for exp in relevant_exps:
                df = load_global_metrics(exp["results_path"])
                if df is None:
                    continue
                if metric not in df.columns:
                    continue

                tmp = df[["round", metric]].copy()
                tmp = tmp.rename(columns={metric: "value"})
                tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
                tmp = tmp.dropna(subset=["round", "value"])
                frames.append(tmp)

            if not frames:
                continue

            merged = pd.concat(frames, ignore_index=True)
            stats = (
                merged.groupby("round", as_index=False)["value"]
                .agg(["mean", "std"])
                .reset_index()
            )

            combined[scenario][agg_method] = stats

    return combined

"""This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""

# =========================
# PLOTEO
# =========================

def sanitize_metric_name(metric: str) -> str:
    """
    Limpia el nombre de la métrica para usarlo en el nombre del archivo.
    """
    return re.sub(r"[^\w\-\.]+", "_", metric.strip())


def plot_metric_for_scenario(
    scenario: str,
    metric: str,
    scenario_data: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Genera y guarda un gráfico para una métrica y escenario.
    """
    plt.figure(figsize=(10, 6))

    plotted_any = False

    for agg_method in ["fedavg", "fedprox", "custom"]:
        if agg_method not in scenario_data:
            continue

        stats = scenario_data[agg_method].sort_values("round")
        if stats.empty:
            continue

        x = stats["round"]
        y = stats["mean"]
        y_std = stats["std"].fillna(0)

        label = DISPLAY_NAMES[agg_method]

        plt.plot(x, y, marker="o", linewidth=2, label=label)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    # plt.title(f"{scenario} - {metric}")
    plt.xlabel("Round", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=20)
    plt.tight_layout()

    filename = f"{scenario}_{sanitize_metric_name(metric)}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[OK] Guardado: {output_path}")


# =========================
# FUNCIÓN PRINCIPAL
# =========================

def generate_global_metric_plots(dataset_name: str, root_dir: Path = EXPERIMENTS_ROOT) -> None:
    """
    Genera todos los plots solicitados para un dataset concreto.
    """
    experiments = collect_experiments(root_dir, dataset_name)

    if not experiments:
        print(f"[INFO] No se encontraron experimentos válidos para el dataset '{dataset_name}'.")
        return

    output_dir = PLOTS_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = get_metrics_to_plot(experiments)

    if not metrics:
        print(f"[INFO] No se encontraron métricas para plotear en el dataset '{dataset_name}'.")
        return

    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Experimentos válidos encontrados: {len(experiments)}")
    print(f"[INFO] Métricas a plotear: {metrics}")

    for metric in metrics:
        grouped = combine_experiments_by_metric(experiments, metric)

        for scenario in ["IID", "Non_IID_1.0", "Non_IID_0.5", "Non_IID_0.1"]:
            scenario_data = grouped.get(scenario, {})
            if not scenario_data:
                continue

            plot_metric_for_scenario(
                scenario=scenario,
                metric=metric,
                scenario_data=scenario_data,
                output_dir=output_dir,
            )

    print(f"[DONE] Plots generados en: {output_dir.resolve()}")


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    # Cambia aquí el dataset que quieres procesar
    dataset = ["cifar10"]  # Puedes añadir más datasets a esta lista
    for dataset_name in dataset:
        generate_global_metric_plots(dataset_name)
