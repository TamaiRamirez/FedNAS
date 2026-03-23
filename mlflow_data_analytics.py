import time
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import mlflow


# ======================
# CONFIG
# ======================
EXPERIMENTS_ROOT = Path("/workspace/flwr-enas-nofix/experiments")
ABLATION_ROOT = Path("/workspace/flwr-enas-nofix/ablation")

MLFLOW_TRACKING_URI = None   # e.g. "sqlite:///mlflow.db"
POLL_INTERVAL = 30           # seconds

# If True, finished/invalid experiments are marked accordingly in MLflow tags
MARK_INVALID_AS_TAG_ONLY = True


# ======================
# METADATA PARSING
# ======================
def parse_metadata_full(exp_dir: Path) -> Dict[str, str]:
    """
    Parses both the header fields and CONFIGURATION block from metadata_experiment.txt.
    """
    metadata_file = exp_dir / "metadata_experiment.txt"
    data: Dict[str, str] = {}

    if not metadata_file.exists():
        return data

    in_configuration = False

    with open(metadata_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            if line.strip() == "CONFIGURATION:":
                in_configuration = True
                continue

            if in_configuration:
                if line.startswith("\t") and ":" in line:
                    key, value = line.strip().split(":", maxsplit=1)
                    data[key.strip()] = value.strip()
            else:
                if ":" in line:
                    key, value = line.split(":", maxsplit=1)
                    data[key.strip()] = value.strip()

    return data


def is_valid_experiment(exp_dir: Path) -> bool:
    metadata = parse_metadata_full(exp_dir)
    return metadata.get("STATUS", "").upper() == "VALID"


def get_source_exp_id(exp_dir: Path) -> str:
    """
    Unique identifier for the source experiment.
    Prefer metadata EXPERIMENT ID, otherwise folder name.
    """
    metadata = parse_metadata_full(exp_dir)
    return metadata.get("EXPERIMENT ID", exp_dir.name)


def infer_study_type(exp_dir: Path) -> str:
    root_name = exp_dir.parent.name.lower()
    if root_name == "ablation":
        return "ablation"
    elif root_name == "experiments":
        return "benchmark"
    return "unknown"


def build_mlflow_experiment_name(exp_dir: Path, metadata: Dict[str, str]) -> str:
    """
    Keeps benchmark and ablation separated in MLflow.
    """
    dataset = metadata.get("DATASET_NAME", "UNKNOWN")
    method = metadata.get("AGG_METHOD", "UNKNOWN")
    study_type = infer_study_type(exp_dir)

    if study_type == "ablation":
        return f"ablation/{dataset}/{method}"
    return f"benchmark/{dataset}"


# ======================
# MLFLOW HELPERS
# ======================
def get_or_create_experiment_by_name(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def get_existing_run(experiment_id: str, source_exp_id: str, study_type: str):
    """
    Returns an existing MLflow run for the same source experiment if present.
    """
    filter_string = (
        f"tags.source_exp_id = '{source_exp_id}' "
        f"and tags.study_type = '{study_type}'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        output_format="list",
    )
    return runs[0] if runs else None


def set_common_tags(exp_dir: Path, metadata: Dict[str, str]):
    study_type = infer_study_type(exp_dir)
    mlflow.set_tag("study_type", study_type)
    mlflow.set_tag("source_exp_id", get_source_exp_id(exp_dir))
    mlflow.set_tag("source_path", str(exp_dir))
    mlflow.set_tag("folder_name", exp_dir.name)
    mlflow.set_tag("status", metadata.get("STATUS", "UNKNOWN"))
    mlflow.set_tag("reason", metadata.get("REASON", "N/A"))
    mlflow.set_tag("dataset", metadata.get("DATASET_NAME", "UNKNOWN"))
    mlflow.set_tag("method", metadata.get("AGG_METHOD", "UNKNOWN"))


# ======================
# PARAMS / METRICS / ARTIFACTS
# ======================
def log_params(metadata: Dict[str, str]):
    """
    Logs configuration values as params.
    Avoids volatile header fields like DATE/STATUS/REASON.
    """
    excluded = {"STATUS", "REASON", "DATE", "EXPERIMENT ID"}
    for k, v in metadata.items():
        if k not in excluded:
            mlflow.log_param(k, str(v))


def log_global_metrics_incremental(exp_dir: Path):
    global_file = exp_dir / "results" / "global_metrics.parquet"
    if not global_file.exists():
        return

    df = pd.read_parquet(global_file)
    if df.empty or "round" not in df.columns:
        return

    df = df.sort_values("round")

    for _, row in df.iterrows():
        step = int(row["round"])
        for col in df.columns:
            if col in {"exp_id", "round"}:
                continue
            value = row[col]
            if pd.notna(value):
                try:
                    mlflow.log_metric(col, float(value), step=step)
                except (TypeError, ValueError):
                    pass


def log_client_aggregates_incremental(exp_dir: Path):
    results_dir = exp_dir / "results"
    client_files = list(results_dir.glob("client_*.parquet"))
    if not client_files:
        return

    dfs: List[pd.DataFrame] = []
    for file in client_files:
        try:
            dfs.append(pd.read_parquet(file))
        except Exception:
            continue

    if not dfs:
        return

    df_all = pd.concat(dfs, ignore_index=True)
    if df_all.empty or "round" not in df_all.columns:
        return

    metric_columns = [
        "val_accuracy",
        "val_loss",
        "val_f1_score",
        "test_accuracy",
        "test_loss",
        "test_f1_score",
        "communication_cost",
        "size_model",
    ]
    metric_columns = [c for c in metric_columns if c in df_all.columns]
    if not metric_columns:
        return

    grouped = df_all.groupby("round").agg({
        col: ["mean", "std", "min", "max"] for col in metric_columns
    })

    for round_idx, row in grouped.iterrows():
        step = int(round_idx)
        for metric in metric_columns:
            for stat in ["mean", "std", "min", "max"]:
                value = row[(metric, stat)]
                if pd.notna(value):
                    try:
                        mlflow.log_metric(f"{metric}_{stat}", float(value), step=step)
                    except (TypeError, ValueError):
                        pass


def log_artifacts(exp_dir: Path):
    results_dir = exp_dir / "results"
    if results_dir.exists():
        mlflow.log_artifacts(str(results_dir), artifact_path="results")

    labels_dir = exp_dir / "label_distributions"
    if labels_dir.exists():
        mlflow.log_artifacts(str(labels_dir), artifact_path="label_distributions")

    metadata_file = exp_dir / "metadata_experiment.txt"
    if metadata_file.exists():
        mlflow.log_artifact(str(metadata_file), artifact_path="metadata")


# ======================
# MAIN IMPORT LOGIC
# ======================
def upsert_experiment_run(exp_dir: Path):
    metadata = parse_metadata_full(exp_dir)
    if not metadata:
        return

    study_type = infer_study_type(exp_dir)
    experiment_name = build_mlflow_experiment_name(exp_dir, metadata)
    experiment_id = get_or_create_experiment_by_name(experiment_name)
    source_exp_id = get_source_exp_id(exp_dir)

    existing_run = get_existing_run(experiment_id, source_exp_id, study_type)

    # If experiment is not valid, optionally mark existing run and skip metric/artifact import
    if not is_valid_experiment(exp_dir):
        if existing_run and MARK_INVALID_AS_TAG_ONLY:
            with mlflow.start_run(run_id=existing_run.info.run_id):
                set_common_tags(exp_dir, metadata)
                mlflow.set_tag("import_state", "skipped_non_valid")
        return

    if existing_run:
        run_id = existing_run.info.run_id
        with mlflow.start_run(run_id=run_id):
            set_common_tags(exp_dir, metadata)
            log_params(metadata)
            log_global_metrics_incremental(exp_dir)
            log_client_aggregates_incremental(exp_dir)
            log_artifacts(exp_dir)
        print(f"🔄 Updated existing run: {exp_dir} -> run_id={run_id}")
    else:
        with mlflow.start_run(run_name=exp_dir.name, experiment_id=experiment_id):
            set_common_tags(exp_dir, metadata)
            log_params(metadata)
            log_global_metrics_incremental(exp_dir)
            log_client_aggregates_incremental(exp_dir)
            log_artifacts(exp_dir)
        print(f"✅ Created new run: {exp_dir}")


# ======================
# DISCOVERY
# ======================
def iter_candidate_experiment_dirs(root_dir: Path):
    if not root_dir.exists():
        return
    for exp_dir in root_dir.iterdir():
        if exp_dir.is_dir():
            yield exp_dir


def scan_all_roots():
    for root in [EXPERIMENTS_ROOT, ABLATION_ROOT]:
        for exp_dir in iter_candidate_experiment_dirs(root):
            try:
                upsert_experiment_run(exp_dir)
            except Exception as e:
                print(f"❌ Error processing {exp_dir}: {e}")


# ======================
# MAIN LOOP
# ======================
def watch():
    while True:
        scan_all_roots()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("🚀 Starting MLflow watcher for benchmark + ablation...")
    watch()