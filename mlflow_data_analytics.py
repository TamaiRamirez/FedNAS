import time
from pathlib import Path
import pandas as pd
import mlflow

# ======================
# CONFIGURACIÓN
# ======================
ROOT_DIR = "/workspace/flwr-enas-nofix/experiments"  # Carpeta donde están los experimentos
MLFLOW_TRACKING_URI = None          # Dejar None usa local
POLL_INTERVAL = 30                  # Segundos entre scans de nuevos experimentos

# ======================
# FUNCIONES DE UTILIDAD
# ======================
def is_valid_experiment(exp_dir: Path) -> bool:
    metadata_file = exp_dir / "metadata_experiment.txt"
    if not metadata_file.exists():
        return False
    with open(metadata_file) as f:
        for line in f:
            if line.startswith("STATUS:"):
                return "VALID" in line
    return False

def parse_metadata(exp_dir: Path) -> dict:
    config = {}
    metadata_file = exp_dir / "metadata_experiment.txt"
    with open(metadata_file) as f:
        for line in f:
            if ":" in line:
                if "\t" in line:
                    key, val = line.strip().split(":", maxsplit=1)
                    config[key.strip()] = val.strip()
    return config

def get_or_create_experiment(dataset_name: str) -> str:
    exp = mlflow.get_experiment_by_name(dataset_name)
    if exp is None:
        return mlflow.create_experiment(dataset_name)
    return exp.experiment_id

# ======================
# LOG MÉTRICAS
# ======================
def log_global_metrics(exp_dir: Path):
    global_file = exp_dir / "results" / "global_metrics.parquet"
    if not global_file.exists():
        return

    df = pd.read_parquet(global_file)
    for _, row in df.iterrows():
        step = int(row["round"])
        for col in df.columns:
            if col not in ["exp_id", "round"]:
                mlflow.log_metric(col, row[col], step=step)

def log_client_aggregates(exp_dir: Path):
    results_dir = exp_dir / "results"
    dfs = []
    for file in results_dir.glob("client_*.parquet"):
        df = pd.read_parquet(file)
        dfs.append(df)
    if not dfs:
        return
    df_all = pd.concat(dfs)
    # Agrupar por round y calcular medias y desviaciones
    grouped = df_all.groupby("round").agg({
        "val_accuracy": ["mean", "std"],
        "test_accuracy": ["mean", "std"]
    })
    for r, row in grouped.iterrows():
        mlflow.log_metric("val_acc_mean", row[("val_accuracy","mean")], r)
        mlflow.log_metric("val_acc_std", row[("val_accuracy","std")], r)
        mlflow.log_metric("test_acc_mean", row[("test_accuracy","mean")], r)
        mlflow.log_metric("test_acc_std", row[("test_accuracy","std")], r)

def log_artifacts(exp_dir: Path):
    # Guardar todos los archivos de interés
    mlflow.log_artifacts(exp_dir / "results", artifact_path="results")
    labels_dir = exp_dir / "label_distributions"
    if labels_dir.exists():
        mlflow.log_artifacts(labels_dir, artifact_path="labels")
    mlflow.log_artifact(exp_dir / "metadata_experiment.txt")

# ======================
# LOG PARAMS
# ======================
def log_params(exp_dir: Path):
    config = parse_metadata(exp_dir)
    for k, v in config.items():
        mlflow.log_param(k, v)

# ======================
# PROCESAR EXPERIMENTO
# ======================
def log_experiment_to_mlflow(exp_dir: Path):
    if not is_valid_experiment(exp_dir):
        return

    config = parse_metadata(exp_dir)
    dataset = config.get("DATASET_NAME", "UNKNOWN")
    exp_id = get_or_create_experiment(dataset)

    with mlflow.start_run(run_name=exp_dir.name, experiment_id=exp_id):
        log_params(exp_dir)
        log_global_metrics(exp_dir)
        log_client_aggregates(exp_dir)
        log_artifacts(exp_dir)
    print(f"✅ Experimento registrado: {exp_dir.name}")

# ======================
# WATCHER EN TIEMPO REAL
# ======================
def watch_experiments(root_dir: str):
    processed = set()
    while True:
        for exp_dir in Path(root_dir).iterdir():
            if not exp_dir.is_dir():
                continue
            if exp_dir in processed:
                continue
            if is_valid_experiment(exp_dir):
                log_experiment_to_mlflow(exp_dir)
                processed.add(exp_dir)
        time.sleep(POLL_INTERVAL)

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("🚀 Lanzando watcher de experimentos MLflow...")
    watch_experiments(ROOT_DIR)
