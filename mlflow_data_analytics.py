import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


# ======================
# CONFIG
# ======================
EXPERIMENTS_ROOT = Path("/workspace/flwr-enas-nofix/experiments")
ABLATION_ROOT = Path("/workspace/flwr-enas-nofix/ablation")

MLFLOW_TRACKING_URI = None
POLL_INTERVAL = 30

DELETE_RUNS_FOR_NON_VALID = False
DELETE_DUPLICATE_RUNS = False


client = MlflowClient()


# ======================
# METADATA PARSING
# ======================
def parse_metadata_full(exp_dir: Path) -> Dict[str, str]:
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


def get_status(exp_dir: Path) -> str:
    metadata = parse_metadata_full(exp_dir)
    return metadata.get("STATUS", "").upper()


def is_valid_experiment(exp_dir: Path) -> bool:
    return get_status(exp_dir) == "VALID"


def get_source_exp_id(exp_dir: Path) -> str:
    metadata = parse_metadata_full(exp_dir)
    return metadata.get("EXPERIMENT ID", exp_dir.name)


def get_source_path(exp_dir: Path) -> str:
    return str(exp_dir.resolve())


def infer_study_type(exp_dir: Path) -> str:
    root_name = exp_dir.parent.name.lower()
    if root_name == "ablation":
        return "ablation"
    if root_name == "experiments":
        return "benchmark"
    return "unknown"


def build_mlflow_experiment_name(exp_dir: Path, metadata: Dict[str, str]) -> str:
    dataset = metadata.get("DATASET_NAME", "UNKNOWN")
    method = metadata.get("AGG_METHOD", "UNKNOWN")
    study_type = infer_study_type(exp_dir)

    if study_type == "ablation":
        return f"ablation/{dataset}/{method}"
    if study_type == "benchmark":
        return f"benchmark/{dataset}"
    return f"other/{dataset}"


# ======================
# MLFLOW HELPERS
# ======================
def get_or_create_experiment_by_name(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def list_all_experiment_ids() -> List[str]:
    exps = mlflow.search_experiments(view_type=1)
    return [e.experiment_id for e in exps]


def find_runs_by_source_path(source_path: str):
    """
    Search across all experiments, including active and deleted runs.
    """
    all_exp_ids = list_all_experiment_ids()
    if not all_exp_ids:
        return []

    filter_string = f"tags.source_path = '{source_path}'"
    runs = mlflow.search_runs(
        experiment_ids=all_exp_ids,
        filter_string=filter_string,
        output_format="list",
    )
    return runs


def find_best_existing_run(source_path: str):
    runs = find_runs_by_source_path(source_path)
    if not runs:
        return None

    # Prefer active runs over deleted, newest first
    runs = sorted(
        runs,
        key=lambda r: (
            0 if r.info.lifecycle_stage == "active" else 1,
            r.info.start_time or 0,
        ),
        reverse=False,
    )
    return runs[0]


def delete_extra_duplicate_runs(source_path: str):
    runs = find_runs_by_source_path(source_path)
    if len(runs) <= 1:
        return

    # Keep first active run, delete the rest
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            0 if r.info.lifecycle_stage == "active" else 1,
            r.info.start_time or 0,
        )
    )
    keep = runs_sorted[0]

    for run in runs_sorted[1:]:
        if run.info.lifecycle_stage != "deleted":
            client.delete_run(run.info.run_id)
            print(f"🗑️ Deleted duplicate run {run.info.run_id} for {source_path}")

    return keep


def set_common_tags(exp_dir: Path, metadata: Dict[str, str]):
    mlflow.set_tag("study_type", infer_study_type(exp_dir))
    mlflow.set_tag("source_exp_id", get_source_exp_id(exp_dir))
    mlflow.set_tag("source_path", get_source_path(exp_dir))
    mlflow.set_tag("folder_name", exp_dir.name)
    mlflow.set_tag("status", metadata.get("STATUS", "UNKNOWN"))
    mlflow.set_tag("reason", metadata.get("REASON", "N/A"))
    mlflow.set_tag("dataset", metadata.get("DATASET_NAME", "UNKNOWN"))
    mlflow.set_tag("method", metadata.get("AGG_METHOD", "UNKNOWN"))


def maybe_move_run_to_correct_experiment(run_id: str, target_experiment_id: str) -> str:
    run = client.get_run(run_id)
    current_experiment_id = run.info.experiment_id
    if current_experiment_id != target_experiment_id:
        client.rename_run(run_id, run.data.tags.get("mlflow.runName", run_id))
        client._tracking_client.store.tracking._get_store()._move_run_to_experiment  # noqa
    return run_id


# ======================
# PARAMS / METRICS / ARTIFACTS
# ======================
def log_params(metadata: Dict[str, str]):
    excluded = {"STATUS", "REASON", "DATE", "EXPERIMENT ID"}
    for k, v in metadata.items():
        if k not in excluded:
            mlflow.log_param(k, str(v))


def log_global_metrics(exp_dir: Path):
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


def log_client_aggregates(exp_dir: Path):
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
# MAIN UPSERT
# ======================
def upsert_experiment_run(exp_dir: Path):
    metadata = parse_metadata_full(exp_dir)
    if not metadata:
        return

    source_path = get_source_path(exp_dir)
    source_exp_id = get_source_exp_id(exp_dir)
    status = get_status(exp_dir)

    target_experiment_name = build_mlflow_experiment_name(exp_dir, metadata)
    target_experiment_id = get_or_create_experiment_by_name(target_experiment_name)

    if DELETE_DUPLICATE_RUNS:
        existing_run = delete_extra_duplicate_runs(source_path)
    else:
        existing_run = find_best_existing_run(source_path)

    # If current folder is not VALID, update or delete any old run and stop
    if status != "VALID":
        if existing_run:
            run_id = existing_run.info.run_id
            if DELETE_RUNS_FOR_NON_VALID:
                if existing_run.info.lifecycle_stage != "deleted":
                    client.delete_run(run_id)
                    print(f"🗑️ Deleted non-VALID run: {exp_dir} -> run_id={run_id}")
            else:
                with mlflow.start_run(run_id=run_id):
                    set_common_tags(exp_dir, metadata)
                    mlflow.set_tag("import_state", "non_valid")
                print(f"⏭️ Marked non-VALID run: {exp_dir} -> run_id={run_id}")
        return

    # VALID experiment
    if existing_run and existing_run.info.lifecycle_stage == "active":
        run_id = existing_run.info.run_id
        with mlflow.start_run(run_id=run_id):
            set_common_tags(exp_dir, metadata)
            log_params(metadata)
            log_global_metrics(exp_dir)
            log_client_aggregates(exp_dir)
            log_artifacts(exp_dir)
            mlflow.set_tag("import_state", "updated")
        print(f"🔄 Updated existing run: {exp_dir} -> run_id={run_id}")
    else:
        with mlflow.start_run(run_name=exp_dir.name, experiment_id=target_experiment_id):
            set_common_tags(exp_dir, metadata)
            log_params(metadata)
            log_global_metrics(exp_dir)
            log_client_aggregates(exp_dir)
            log_artifacts(exp_dir)
            mlflow.set_tag("import_state", "created")
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