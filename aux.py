from pathlib import Path
import numpy as np
import pandas as pd


EPS = 1e-12


def compute_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return float(np.std(values) / (np.mean(values) + EPS))


def compute_jain_index(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return float((np.sum(values) ** 2) / (len(values) * np.sum(values ** 2) + EPS))


def load_all_clients_metrics(results_dir: Path) -> pd.DataFrame:
    client_files = sorted(results_dir.glob("client_*.parquet"))
    if not client_files:
        raise FileNotFoundError(f"No se encontraron archivos client_*.parquet en {results_dir}")

    dfs = []
    for file in client_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    clients_df = pd.concat(dfs, ignore_index=True)

    required_cols = {"round", "val_f1_score"}
    missing = required_cols - set(clients_df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas necesarias en los client_*.parquet de {results_dir}: {missing}"
        )

    return clients_df


def compute_valid_fairness_by_round(clients_df: pd.DataFrame) -> pd.DataFrame:
    fairness_df = (
        clients_df.groupby("round")["val_f1_score"]
        .apply(lambda s: pd.Series({
            "CV_valid": compute_cv(s.to_numpy()),
            "Jain_index_valid": compute_jain_index(s.to_numpy()),
        }))
        .unstack()
        .reset_index()
    )

    return fairness_df


def update_global_metrics(results_dir: Path, backup: bool = True) -> None:
    global_file = results_dir / "global_metrics.parquet"
    if not global_file.exists():
        raise FileNotFoundError(f"No existe {global_file}")

    clients_df = load_all_clients_metrics(results_dir)
    fairness_valid_df = compute_valid_fairness_by_round(clients_df)

    global_df = pd.read_parquet(global_file)

    if "round" not in global_df.columns:
        raise ValueError(f"El archivo {global_file} no tiene columna 'round'")

    # Renombrar métricas antiguas si todavía no se han renombrado
    rename_map = {}
    if "CV" in global_df.columns and "CV_test" not in global_df.columns:
        rename_map["CV"] = "CV_test"
    if "Jain_index" in global_df.columns and "Jain_index_test" not in global_df.columns:
        rename_map["Jain_index"] = "Jain_index_test"

    if rename_map:
        global_df = global_df.rename(columns=rename_map)

    # Eliminar columnas si ya existían para recalcular limpio
    for col in ["CV_valid", "Jain_index_valid"]:
        if col in global_df.columns:
            global_df = global_df.drop(columns=[col])

    # Merge por round
    updated_df = global_df.merge(fairness_valid_df, on="round", how="left")

    # Orden opcional de columnas
    preferred_order = [
        "exp_id",
        "round",
        "loss",
        "CV_valid",
        "Jain_index_valid",
        "CV_test",
        "Jain_index_test",
    ]

    existing_preferred = [c for c in preferred_order if c in updated_df.columns]
    remaining_cols = [c for c in updated_df.columns if c not in existing_preferred]
    updated_df = updated_df[existing_preferred + remaining_cols]

    if backup:
        backup_file = results_dir / "global_metrics_backup.parquet"
        global_df.to_parquet(backup_file, index=False)

    updated_df.to_parquet(global_file, index=False)


def process_experiments(base_dir: str, experiment_ids: list[str], backup: bool = True) -> None:
    base_path = Path(base_dir)

    for exp_id in experiment_ids:
        results_dir = base_path / exp_id / "results"
        try:
            print(f"[INFO] Procesando experimento: {exp_id}")
            update_global_metrics(results_dir, backup=backup)
            print(f"[OK] Actualizado: {results_dir / 'global_metrics.parquet'}")
        except Exception as e:
            print(f"[ERROR] Experimento {exp_id}: {e}")


if __name__ == "__main__":
    BASE_DIR = "/workspace/flwr-enas-nofix/experiments"
    EXPERIMENTS = ["20260324_1", "20260325_1", "20260325_2", "20260326_1", "20260326_2"]

    process_experiments(BASE_DIR, EXPERIMENTS, backup=True)