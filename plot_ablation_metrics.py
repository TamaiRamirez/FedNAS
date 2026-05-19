#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera plots de métricas globales de experimentos de Federated Learning.

Uso:
    python plot_global_metrics_by_beta.py /ruta/al/directorio_raiz_experimentos

Estructura esperada por experimento:
    YYYYMMDD_X/
        metadata_experiment.txt
        results/
            global_metrics.parquet

Salida:
    /ruta/al/directorio_raiz_experimentos/Plots/
        metric_loss.png
        metric_CV_valid.png
        metric_Jain_index_valid.png
        ...

Notas:
- Se genera un gráfico por métrica.
- En cada gráfico aparece una línea por valor de BETA.
- Si hay varios experimentos con el mismo BETA, se promedian por ronda.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR_PATTERN = re.compile(r"^\d{8}_\d+$")


def is_experiment_dir(path: Path) -> bool:
    """Comprueba si el nombre del directorio tiene formato YYYYMMDD_X."""
    return path.is_dir() and bool(EXPERIMENT_DIR_PATTERN.match(path.name))


def parse_metadata_file(metadata_path: Path) -> Dict[str, str]:
    """
    Parseo simple de metadata_experiment.txt.
    Extrae pares tipo KEY: VALUE.
    También soporta líneas con indentación.
    """
    metadata: Dict[str, str] = {}

    if not metadata_path.exists():
        return metadata

    with metadata_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            metadata[key] = value

    return metadata


def extract_beta(metadata: Dict[str, str]) -> Optional[float]:
    """
    Extrae el valor de BETA desde metadata.
    Busca primero una clave exacta 'BETA'.
    Luego intenta localizar patrones como 'BETA=0.5' o 'BETA: 0.5'
    en cualquier valor del metadata.
    """
    if "BETA" in metadata:
        try:
            return float(metadata["BETA"])
        except ValueError:
            pass

    beta_regex = re.compile(r"\bBETA\b\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    for value in metadata.values():
        match = beta_regex.search(value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def sanitize_filename(name: str) -> str:
    """Limpia un nombre para usarlo como parte de un nombre de archivo."""
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", name)


def load_experiment_data(exp_dir: Path) -> Optional[Tuple[float, pd.DataFrame, str]]:
    """
    Carga metadata y parquet de un experimento.
    Devuelve:
        (beta, dataframe, exp_id_dirname)
    o None si no se puede usar.
    """
    metadata_path = exp_dir / "metadata_experiment.txt"
    parquet_path = exp_dir / "results" / "global_metrics.parquet"

    if not metadata_path.exists():
        print(f"[WARN] No existe metadata_experiment.txt en: {exp_dir}")
        return None

    if not parquet_path.exists():
        print(f"[WARN] No existe global_metrics.parquet en: {parquet_path}")
        return None

    metadata = parse_metadata_file(metadata_path)
    beta = extract_beta(metadata)

    if beta is None:
        print(f"[WARN] No se encontró BETA en metadata_experiment.txt de: {exp_dir}")
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[WARN] No se pudo leer parquet en {parquet_path}: {e}")
        return None

    if "round" not in df.columns:
        print(f"[WARN] El parquet no contiene la columna 'round' en: {parquet_path}")
        return None

    return beta, df, exp_dir.name


def collect_all_experiments(root_dir: Path) -> List[Tuple[float, pd.DataFrame, str]]:
    """Recorre todos los experimentos válidos bajo root_dir."""
    experiments: List[Tuple[float, pd.DataFrame, str]] = []

    for item in sorted(root_dir.iterdir()):
        if not is_experiment_dir(item):
            continue

        loaded = load_experiment_data(item)
        if loaded is not None:
            experiments.append(loaded)

    return experiments


def get_metric_columns(experiments: List[Tuple[float, pd.DataFrame, str]]) -> List[str]:
    """
    Obtiene el conjunto de métricas ploteables.
    Excluye columnas no numéricas y columnas auxiliares típicas.
    """
    exclude = {"exp_id", "round"}

    metric_cols = set()
    for _, df, _ in experiments:
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                metric_cols.add(col)

    return sorted(metric_cols)


def build_metric_dataframe(
    experiments: List[Tuple[float, pd.DataFrame, str]],
    metric: str
) -> pd.DataFrame:
    """
    Construye un DataFrame unificado con columnas:
        beta, round, value, exp_name

    Solo incluye filas válidas para la métrica indicada.
    """
    rows = []

    for beta, df, exp_name in experiments:
        if metric not in df.columns:
            continue

        subset = df[["round", metric]].copy()
        subset = subset.rename(columns={metric: "value"})
        subset["beta"] = beta
        subset["exp_name"] = exp_name

        # Filtrar valores no numéricos o NaN
        subset = subset[pd.to_numeric(subset["value"], errors="coerce").notna()].copy()
        subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
        subset["round"] = pd.to_numeric(subset["round"], errors="coerce")
        subset = subset.dropna(subset=["round", "value"])

        rows.append(subset)

    if not rows:
        return pd.DataFrame(columns=["beta", "round", "value", "exp_name"])

    return pd.concat(rows, ignore_index=True)


def plot_metric_by_beta(
    metric_df: pd.DataFrame,
    metric_name: str,
    output_dir: Path
) -> None:
    """
    Genera un gráfico por métrica con una línea por BETA.
    Si hay varios experimentos con el mismo BETA, se promedian por ronda.
    """
    if metric_df.empty:
        print(f"[INFO] No hay datos para la métrica: {metric_name}")
        return

    # Promedio por beta y ronda
    grouped = (
        metric_df.groupby(["beta", "round"], as_index=False)["value"]
        .mean()
        .sort_values(["beta", "round"])
    )

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    for beta_value in sorted(grouped["beta"].unique()):
        beta_data = grouped[grouped["beta"] == beta_value].sort_values("round")
        ax.plot(
            beta_data["round"],
            beta_data["value"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=rf"$\beta={beta_value:g}$",
        )

    # plt.title(f"Evolution of {metric_name} by round for different BETA values")
    ax.set_xlabel("Round", fontsize=24)
    ax.set_ylabel("Value", fontsize=24)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        fontsize=22,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )

    output_file = output_dir / f"metric_{sanitize_filename(metric_name)}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Plot guardado: {output_file}")

"""This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Genera plots de métricas globales para experimentos de ablación."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Directorio raíz de los experimentos de ablación.",
    )
    args = parser.parse_args()

    if args.root_dir is not None:
        root_dir = Path(args.root_dir).expanduser().resolve()
    else:
        root_dir = (Path(__file__).resolve().parent / "flwr-enas-nofix" / "ablation").resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[ERROR] Ruta no válida: {root_dir}")
        sys.exit(1)

    output_dir = root_dir / "Plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = collect_all_experiments(root_dir)

    if not experiments:
        print("[ERROR] No se encontraron experimentos válidos con BETA y global_metrics.parquet.")
        sys.exit(1)

    metric_columns = get_metric_columns(experiments)

    if not metric_columns:
        print("[ERROR] No se encontraron métricas numéricas para plotear.")
        sys.exit(1)

    print(f"[INFO] Experimentos válidos encontrados: {len(experiments)}")
    print(f"[INFO] Métricas detectadas: {', '.join(metric_columns)}")

    for metric in metric_columns:
        metric_df = build_metric_dataframe(experiments, metric)
        plot_metric_by_beta(metric_df, metric, output_dir)

    print(f"[DONE] Todos los plots se han guardado en: {output_dir}")


if __name__ == "__main__":
    main()
