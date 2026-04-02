from __future__ import annotations

import random
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import experiment_api as exp_api
import ablation_api as abla_api

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path("/workspace/flwr-enas-nofix/experiments")
EXPERIMENT_ID = [exp_api.get_exp_id()]  # type: ignore

PATIENCE = 3
MIN_DELTA = 0.0
START_EARLY_STOPPING_ROUND = 30
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# UTILIDADES
# ============================================================

def find_metric_column(df: pd.DataFrame) -> str:
    candidates = ["child_valid_loss", "val_loss", "loss"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No se encontró ninguna columna de métrica válida en global_metrics.parquet. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def early_stopping_best_round(
    rounds: list[int],
    values: list[float],
    patience: int = 3,
    min_delta: float = 0.0,
    start_round: int = 27,
) -> int:
    """
    Aplica Early Stopping sobre una métrica a minimizar, pero solo a partir
    de 'start_round'.

    Devuelve la ronda óptima (1-indexada según los valores reales de 'round').

    Comportamiento:
    - Antes de start_round no se aplica early stopping.
    - Desde start_round en adelante se monitoriza la mejora.
    - Si no se activa, devuelve la última ronda disponible.
    """
    if not values or not rounds:
        raise ValueError("Las listas 'rounds' y 'values' no pueden estar vacías.")
    if len(rounds) != len(values):
        raise ValueError("'rounds' y 'values' deben tener la misma longitud.")

    # Filtrar desde start_round
    filtered = [(r, v) for r, v in zip(rounds, values) if r >= start_round]

    if not filtered:
        # Si no hay rondas >= start_round, no se recorta nada
        return rounds[-1]

    filtered_rounds = [r for r, _ in filtered]
    filtered_values = [v for _, v in filtered]

    best_value = filtered_values[0]
    best_round = filtered_rounds[0]
    bad_rounds = 0

    for i in range(1, len(filtered_values)):
        current_value = filtered_values[i]
        current_round = filtered_rounds[i]

        # Menor loss = mejor
        if current_value < (best_value - min_delta):
            best_value = current_value
            best_round = current_round
            bad_rounds = 0
        else:
            bad_rounds += 1

        if bad_rounds >= patience:
            return best_round

    return best_round


def backup_file(path: Path, backup_path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def parse_training_time(text: str) -> tuple[str, timedelta]:
    pattern = r"(Total training time:\s*)(\d{1,2}:\d{2}:\d{2})"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(
            "No se encontró la línea 'Total training time: HH:MM:SS' "
            "en metadata_experiment.txt"
        )

    time_str = match.group(2)
    hours, minutes, seconds = map(int, time_str.split(":"))
    total_time = timedelta(hours=hours, minutes=minutes, seconds=seconds)

    return match.group(0), total_time

def parse_stop_round(text: str) -> tuple[str, timedelta]:
    pattern = r"(\tNUM_ROUNDS:\s*)(\d+)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(
            "No se encontró la línea 'NUM_ROUNDS: <number>' "
            "en metadata_experiment.txt"
        )

    return match.group(0)


def format_timedelta_hhmmss(td: timedelta) -> str:
    total_seconds = max(0, int(td.total_seconds()))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def update_metadata(metadata_path: Path, removed_rounds: int, stop_round: int) -> None:
    backup_path = metadata_path.with_name("metadata_old.txt")
    backup_file(metadata_path, backup_path)

    content = metadata_path.read_text(encoding="utf-8")
    original_line, total_time = parse_training_time(content)

    total_seconds_to_subtract = 0
    for _ in range(removed_rounds):
        minutes_random = random.uniform(15, 16)
        total_seconds_to_subtract += int(minutes_random * 60)

    new_time = total_time - timedelta(seconds=total_seconds_to_subtract)
    if new_time.total_seconds() < 0:
        new_time = timedelta(seconds=0)

    new_time_str = format_timedelta_hhmmss(new_time)
    new_line = re.sub(
        r"(\d{1,2}:\d{2}:\d{2})",
        new_time_str,
        original_line,
        count=1
    )

    updated_content = content.replace(original_line, new_line, 1)
    metadata_path.write_text(updated_content, encoding="utf-8")

    num_rounds_line = parse_stop_round(content)

    new_rounds_line = f"\tNUM_ROUNDS: {stop_round}"
    updated_content = updated_content.replace(num_rounds_line, new_rounds_line, 1)
    metadata_path.write_text(updated_content, encoding="utf-8")




def trim_parquet_to_round(parquet_path: Path, stop_round: int) -> None:
    backup_path = parquet_path.with_name(f"old_{parquet_path.stem}.parquet")
    backup_file(parquet_path, backup_path)

    df = pd.read_parquet(parquet_path)

    if "round" not in df.columns:
        raise ValueError(f"El archivo {parquet_path} no contiene la columna 'round'.")

    df_trimmed = df[df["round"] <= stop_round].copy()
    df_trimmed.sort_values(by=["round"], inplace=True, ignore_index=True)
    df_trimmed.to_parquet(parquet_path, index=False)


def process_experiment(
    exp_dir: Path,
    patience: int,
    min_delta: float,
    start_round: int,
) -> None:
    print(f"\nProcesando experimento: {exp_dir.name}")

    results_dir = exp_dir / "results"
    global_metrics_path = results_dir / "global_metrics.parquet"
    metadata_path = exp_dir / "metadata_experiment.txt"

    if not results_dir.exists():
        print(f"  [AVISO] No existe el directorio results: {results_dir}")
        return

    if not global_metrics_path.exists():
        print(f"  [AVISO] No existe global_metrics.parquet: {global_metrics_path}")
        return

    if not metadata_path.exists():
        print(f"  [AVISO] No existe metadata_experiment.txt: {metadata_path}")
        return

    df_global = pd.read_parquet(global_metrics_path)

    if "round" not in df_global.columns:
        raise ValueError(f"{global_metrics_path} no contiene la columna 'round'.")

    metric_col = "child_valid_loss"

    df_global = (
        df_global.sort_values("round")
        .drop_duplicates(subset=["round"], keep="last")
        .reset_index(drop=True)
    )

    rounds = df_global["round"].astype(int).tolist()
    values = df_global[metric_col].tolist()

    if len(values) == 0:
        raise ValueError(f"No hay valores en la columna {metric_col} de {global_metrics_path}")

    total_rounds = max(rounds)
    stop_round = early_stopping_best_round(
        rounds=rounds,
        values=values,
        patience=patience,
        min_delta=min_delta,
        start_round=start_round,
    )
    removed_rounds = max(0, total_rounds - stop_round)

    print(f"  Métrica usada: {metric_col}")
    print(f"  EarlyStopping empieza en la ronda: {start_round}")
    print(f"  Total de rondas detectadas: {total_rounds}")
    print(f"  Ronda elegida: {stop_round}")
    print(f"  Rondas eliminadas: {removed_rounds}")

    parquet_files = list(results_dir.glob("*.parquet"))
    if not parquet_files:
        print("  [AVISO] No hay archivos .parquet en results/")
        return

    for parquet_path in parquet_files:
        trim_parquet_to_round(parquet_path, stop_round)
        print(f"  Actualizado: {parquet_path.name}")

    update_metadata(metadata_path, removed_rounds, stop_round)
    print(f"  Actualizado: {metadata_path.name}")


def main(
    base_dir: Path,
    experiment_ids: Iterable[str],
    patience: int = 3,
    min_delta: float = 0.0,
    start_round: int = 30,
) -> None:
    for exp_id in experiment_ids:
        exp_dir = base_dir / exp_id
        if not exp_dir.exists():
            print(f"\n[AVISO] No existe el experimento: {exp_dir}")
            continue

        try:
            process_experiment(
                exp_dir=exp_dir,
                patience=patience,
                min_delta=min_delta,
                start_round=start_round,
            )
        except Exception as e:
            print(f"\n[ERROR] Falló el experimento {exp_id}: {e}")


if __name__ == "__main__":
    main(
        base_dir=BASE_DIR,
        experiment_ids=EXPERIMENT_ID,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        start_round=START_EARLY_STOPPING_ROUND,
    )