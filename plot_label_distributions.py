from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS_ROOT = Path("/workspace/flwr-enas-nofix/experiments")


def parse_metadata(metadata_path: Path) -> Dict[str, str]:
    """
    Parses metadata.txt into a simple dictionary.

    Expected lines include:
        STATUS: VALID
        DATASET_NAME: ...
        DATA PARTITIONING: IID
        DATA PARTITIONING: Non-IID with alpha=0.5
        AGG_METHOD: fedavg
    """
    metadata: Dict[str, str] = {}

    with metadata_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata


def scenario_from_partitioning(data_partitioning: str) -> str:
    """
    Converts DATA PARTITIONING metadata into the requested scenario name.

    Examples
    --------
    IID -> IID
    Non-IID with alpha=1.0 -> Non-IID_1.0
    Non-IID with alpha=0.5 -> Non-IID_0.5
    """
    if data_partitioning == "IID":
        return "IID"

    match = re.search(r"alpha\s*=\s*([0-9.]+)", data_partitioning)

    if match:
        alpha = match.group(1)
        return f"Non-IID_{alpha}"

    raise ValueError(f"Could not infer scenario from DATA PARTITIONING: {data_partitioning}")


def load_label_distributions(label_dist_dir: Path) -> pd.DataFrame:
    """
    Loads all client_*.parquet files from label_distributions.
    """
    if not label_dist_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {label_dist_dir}")

    parquet_files = sorted(label_dist_dir.glob("client_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No se encontraron ficheros client_*.parquet en {label_dist_dir}"
        )

    dfs: List[pd.DataFrame] = []

    for pq in parquet_files:
        df = pd.read_parquet(pq)

        required_cols = {"partition_id", "split", "class", "count"}
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"Faltan columnas {missing} en {pq}")

        df = df.copy()
        df["partition_id"] = df["partition_id"].astype(int)
        df["class"] = df["class"].astype(int)
        df["count"] = df["count"].astype(int)
        df["split"] = df["split"].astype(str)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_pivot(
    data: pd.DataFrame,
    split: str,
    clients: List[int],
    classes: List[int],
    normalize: bool,
) -> pd.DataFrame:
    """
    Builds a client x class pivot table for a given split.
    """
    split_df = data[data["split"] == split].copy()

    pivot = (
        split_df.pivot_table(
            index="partition_id",
            columns="class",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=clients, columns=classes, fill_value=0)
        .astype(float)
    )

    if normalize:
        row_sums = pivot.sum(axis=1).replace(0, 1.0)
        pivot = pivot.div(row_sums, axis=0)

    return pivot


def plot_train_valid_shared_heatmap(
    data: pd.DataFrame,
    output_path: Path,
    title: Optional[str] = None,
    normalize: bool = False,
    dpi: int = 600,
    cmap: str = "coolwarm",
) -> tuple[float, float]:
    """
    Creates one heatmap with a single generic colorbar.

    The heatmap values are percentages of the total number of examples
    in the split. The colorbar uses the actual min and max percentages.

    Returns
    -------
    tuple[float, float]
        train_total_examples, valid_total_examples
    """
    available_splits = set(data["split"].unique())

    if "train" not in available_splits or "valid" not in available_splits:
        raise ValueError(
            f"Expected both train and valid splits, found: {sorted(available_splits)}"
        )

    clients = sorted(data["partition_id"].unique().tolist())
    classes = sorted(data["class"].unique().tolist())

    train_pivot = build_pivot(
        data=data,
        split="train",
        clients=clients,
        classes=classes,
        normalize=False,
    )

    valid_pivot = build_pivot(
        data=data,
        split="valid",
        clients=clients,
        classes=classes,
        normalize=False,
    )

    train_total_examples = float(train_pivot.values.sum()) / len(clients)
    valid_total_examples = float(valid_pivot.values.sum()) / len(clients)

    if train_total_examples <= 0:
        raise ValueError("Train total number of examples must be greater than zero.")

    if valid_total_examples <= 0:
        raise ValueError("Valid total number of examples must be greater than zero.")

    # Plot the train split as percentage of total train examples.
    # Since train and valid have the same distribution pattern, this same
    # visual pattern represents both splits, while totals are printed separately.
    heatmap_values = (train_pivot.values / train_total_examples) * 100.0

    vmin = float(heatmap_values.min())
    vmax = float(heatmap_values.max())

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig_w = max(7, min(16, 0.10 * len(classes) + 6))
    fig_h = max(3.5, 0.75 * len(clients) + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        heatmap_values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    ax.grid(False)

    ax.set_yticks(np.arange(len(clients)))
    ax.set_yticklabels([f"{c}" for c in clients], fontsize=18)

    tick_step = compute_tick_step(len(classes))
    tick_idx = np.arange(0, len(classes), tick_step)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([classes[j] for j in tick_idx], rotation=0, fontsize=20)

    ax.set_xlabel("Class", fontsize=22)
    ax.set_ylabel("Client", fontsize=22)

    # if title is None:
    #     title = "Class distribution heatmap"

    # ax.set_title(title, fontsize=12)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Total examples (%)", fontsize=20)

    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks], rotation=0, fontsize=16)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return train_total_examples, valid_total_examples

"""This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""

def compute_tick_step(n_classes: int) -> int:
    """
    Reduces the number of x-axis labels when there are many classes.
    """
    if n_classes <= 10:
        return 1
    if n_classes <= 20:
        return 2
    if n_classes <= 50:
        return 5
    if n_classes <= 100:
        return 10

    return max(1, n_classes // 10)


def should_process_experiment(metadata: Dict[str, str]) -> bool:
    """
    Returns True only for experiments with:
        STATUS == VALID
        AGG_METHOD == fedavg
    """
    status = metadata.get("STATUS", "").strip().lower()
    agg_method = metadata.get("AGG_METHOD", "").strip().lower()

    return status == "valid" and agg_method == "fedavg"


def process_valid_fedavg_experiments(
    experiments_root: str | Path = EXPERIMENTS_ROOT,
    normalize: bool = False,
    dpi: int = 400,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    """
    Scans the experiments directory and generates merged train/valid heatmaps
    only for VALID fedavg experiments.

    Output:
        experiments_root.parent/Plots/<DATASET_NAME>/<scenario>_data_distribution.png
    """
    experiments_root = Path(experiments_root)

    if not experiments_root.exists():
        raise FileNotFoundError(f"No existe el directorio: {experiments_root}")

    plt.style.use(style)

    experiment_dirs = sorted(
        p
        for p in experiments_root.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{8}_\d+", p.name)
    )

    if not experiment_dirs:
        raise FileNotFoundError(
            f"No se encontraron directorios con formato YYYYMMDD_X en {experiments_root}"
        )

    generated = 0
    skipped = 0

    for experiment_dir in experiment_dirs:
        metadata_path = experiment_dir / "metadata_experiment.txt"

        if not metadata_path.exists():
            skipped += 1
            print(f"[SKIP] {experiment_dir.name}: metadata_experiment.txt not found")
            continue

        metadata = parse_metadata(metadata_path)

        if not should_process_experiment(metadata):
            skipped += 1
            print(f"[SKIP] {experiment_dir.name}: not VALID + fedavg")
            continue

        dataset_name = metadata.get("DATASET_NAME")
        data_partitioning = metadata.get("DATA PARTITIONING")

        if not dataset_name or not data_partitioning:
            skipped += 1
            print(
                f"[SKIP] {experiment_dir.name}: missing DATASET_NAME or DATA PARTITIONING"
            )
            continue

        try:
            scenario = scenario_from_partitioning(data_partitioning)
        except ValueError as exc:
            skipped += 1
            print(f"[SKIP] {experiment_dir.name}: {exc}")
            continue

        label_dist_dir = experiment_dir / "label_distributions"

        try:
            data = load_label_distributions(label_dist_dir)
        except Exception as exc:
            skipped += 1
            print(f"[SKIP] {experiment_dir.name}: could not load label distributions: {exc}")
            continue

        output_dir = experiments_root.parent / "Plots" / dataset_name
        output_path = output_dir / f"{scenario}_data_distribution.png"

        title = f"Class distribution heatmap"

        try:
            train_total, valid_total = plot_train_valid_shared_heatmap(
                data=data,
                output_path=output_path,
                #title=title,
                normalize=normalize,
                dpi=dpi,
            )

            print(
                f"[OK] {experiment_dir.name} -> {output_path} | "
                f"train total examples = {train_total:.0f}, "
                f"valid total examples = {valid_total:.0f}"
            )

        except Exception as exc:
            skipped += 1
            print(f"[SKIP] {experiment_dir.name}: could not plot: {exc}")
            continue

        generated += 1

    print()
    print(f"Generated plots: {generated}")
    print(f"Skipped experiments: {skipped}")


if __name__ == "__main__":
    process_valid_fedavg_experiments(
        experiments_root="/workspace/flwr-enas-nofix/experiments",
        normalize=False,
        dpi=600,
    )