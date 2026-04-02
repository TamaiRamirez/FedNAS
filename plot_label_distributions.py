from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_label_distributions_for_experiment(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    clients_per_row: int = 3,
    figsize_per_subplot: tuple[float, float] = (4.2, 2.8),
    dpi: int = 300,
    style: str = "seaborn-v0_8-whitegrid",
    normalize: bool = False,
) -> None:
    """
    Genera visualizaciones de distribución de clases a partir de los parquet
    contenidos en experiment_dir/label_distributions/client_X.parquet.

    Espera columnas:
        - partition_id
        - split        ("train", "valid", "test")
        - class
        - count

    Parámetros
    ----------
    experiment_dir : str | Path
        Ruta al directorio del experimento.
    output_dir : str | Path | None
        Directorio donde guardar las figuras. Si es None, usa label_distributions.
    clients_per_row : int
        Número de subplots por fila.
    figsize_per_subplot : tuple[float, float]
        Tamaño base por subplot.
    dpi : int
        Resolución de guardado.
    style : str
        Estilo de matplotlib.
    normalize : bool
        Si True, representa proporciones en lugar de conteos absolutos.
        Útil para comparar sesgo de clases independientemente del tamaño local.
    """
    experiment_dir = Path(experiment_dir)
    label_dist_dir = experiment_dir / "label_distributions"

    if output_dir is None:
        output_dir = label_dist_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not label_dist_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {label_dist_dir}")

    parquet_files = sorted(label_dist_dir.glob("client_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No se encontraron ficheros client_*.parquet en {label_dist_dir}"
        )

    # Leer y concatenar todos los clientes
    dfs: List[pd.DataFrame] = []
    for pq in parquet_files:
        df = pd.read_parquet(pq)

        required_cols = {"partition_id", "split", "class", "count"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas {missing} en {pq}")

        # Asegurar tipos
        df = df.copy()
        df["partition_id"] = df["partition_id"].astype(int)
        df["class"] = df["class"].astype(int)
        df["count"] = df["count"].astype(int)
        df["split"] = df["split"].astype(str)

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    plt.style.use(style)

    for split in sorted(data["split"].unique()):
        split_df = data[data["split"] == split].copy()
        if split_df.empty:
            continue

        # _plot_per_client_bar_grid(
        #     split_df=split_df,
        #     split=split,
        #     output_dir=output_dir,
        #     clients_per_row=clients_per_row,
        #     figsize_per_subplot=figsize_per_subplot,
        #     dpi=dpi,
        #     normalize=normalize,
        # )

        _plot_heatmap(
            split_df=split_df,
            split=split,
            output_dir=output_dir,
            dpi=dpi,
            normalize=normalize,
        )

    print(f"Figuras guardadas en: {output_dir.resolve()}")


def _plot_per_client_bar_grid(
    split_df: pd.DataFrame,
    split: str,
    output_dir: Path,
    clients_per_row: int,
    figsize_per_subplot: tuple[float, float],
    dpi: int,
    normalize: bool,
) -> None:
    """
    Rejilla 2D de barras por cliente.
    """
    clients = sorted(split_df["partition_id"].unique().tolist())
    classes = sorted(split_df["class"].unique().tolist())

    # Pivot: filas=cliente, columnas=clase
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

    n_clients = len(clients)
    ncols = min(clients_per_row, n_clients)
    nrows = math.ceil(n_clients / ncols)

    fig_w = ncols * figsize_per_subplot[0]
    fig_h = nrows * figsize_per_subplot[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    x = np.arange(len(classes))

    for ax in axes.flat:
        ax.set_visible(False)

    for i, client_id in enumerate(clients):
        ax = axes.flat[i]
        ax.set_visible(True)

        y = pivot.loc[client_id].values
        ax.bar(x, y, width=0.85)

        ax.set_title(f"Client {client_id}", fontsize=10)
        ax.set_xlim(-0.5, len(classes) - 0.5)

        # Reducir densidad de ticks para muchas clases
        tick_step = _compute_tick_step(len(classes))
        tick_idx = np.arange(0, len(classes), tick_step)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([classes[j] for j in tick_idx], rotation=90, fontsize=7)

        if normalize:
            ax.set_ylabel("Proportion", fontsize=8)
        else:
            ax.set_ylabel("Count", fontsize=8)

        ax.set_xlabel("Class", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    title_suffix = "normalized" if normalize else "counts"
    fig.suptitle(
        f"Class distribution per client ({split})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    png_path = output_dir / f"class_distribution_{title_suffix}_{split}.png"
    pdf_path = output_dir / f"class_distribution_{title_suffix}_{split}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    split_df: pd.DataFrame,
    split: str,
    output_dir: Path,
    dpi: int,
    normalize: bool,
) -> None:
    """
    Heatmap cliente × clase, más compacto para datasets con muchas clases.
    Colores: azul (menos ejemplos) -> rojo (más ejemplos).
    """
    clients = sorted(split_df["partition_id"].unique().tolist())
    classes = sorted(split_df["class"].unique().tolist())

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

    if split in ["train", "valid"]:

        if normalize:
            row_sums = pivot.sum(axis=1).replace(0, 1.0)
            pivot = pivot.div(row_sums, axis=0)

        # Tamaño adaptativo
        fig_w = max(6, min(14, 0.12 * len(classes) + 3))
        fig_h = max(3.2, 0.75 * len(clients) + 1.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",   # azul -> rojo
            origin="upper"
        )

        ax.set_yticks(np.arange(len(clients)))
        ax.set_yticklabels([f"Client {c}" for c in clients], fontsize=8)

        tick_step = _compute_tick_step(len(classes))
        tick_idx = np.arange(0, len(classes), tick_step)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([classes[j] for j in tick_idx], rotation=90, fontsize=7)

        ax.set_xlabel("Class", fontsize=9)
        ax.set_ylabel("Client", fontsize=9)
        ax.set_title(f"Class distribution heatmap ({split})", fontsize=11)

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Proportion" if normalize else "Number of Examples", fontsize=8)

        fig.tight_layout()

        title_suffix = "normalized" if normalize else "counts"
        png_path = output_dir / f"class_distribution_heatmap_{title_suffix}_{split}.png"
        pdf_path = output_dir / f"class_distribution_heatmap_{title_suffix}_{split}.pdf"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)


def _compute_tick_step(n_classes: int) -> int:
    """
    Reduce el número de etiquetas en el eje X para que no se solapen.
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


if __name__ == "__main__":
    # Ejemplo de uso:
    # experiment_dir = "/ruta/a/mi_experimento"
    
    
    for exp_dir in ["20260324_1", "20260325_1", "20260325_2","20260326_1"]:
        plot_label_distributions_for_experiment(
            experiment_dir="/workspace/flwr-enas-nofix/experiments/"+exp_dir,
            output_dir=None,      # guarda dentro de label_distributions
            clients_per_row=3,
            figsize_per_subplot=(6, 3),
            dpi=600,
            normalize=False,      # True si prefieres proporciones
        )