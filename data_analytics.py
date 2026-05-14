import pandas as pd

# Ruta del archivo parquet
ruta_archivo = "/workspace/flwr-enas-nofix/experiments/20260305_1/results/global_metrics.parquet"
# Leer archivo parquet
df = pd.read_parquet(ruta_archivo)


# Opcional: mostrar solo las primeras filas
print("\nUltimas 5 filas:")
print(df.tail())

"""
Label distribution analysis & visualization
===========================================

Expected Parquet schema (long / tidy format):
split | class | count | partition_id | exp_id | dataset_name
"""

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ---------------------------------------------------------------------
# # Load label distribution data
# # ---------------------------------------------------------------------

# LABEL_DIST_DIR = Path("path/to/label_distributions")

# df = pd.concat(
#     pd.read_parquet(p)
#     for p in LABEL_DIST_DIR.glob("client_*.parquet")
# ).reset_index(drop=True)

# print("Loaded rows:", len(df))
# print(df.head())

# # ---------------------------------------------------------------------
# # 1. Class distribution per split (per client)
# # ---------------------------------------------------------------------

# for client_id in sorted(df["partition_id"].unique()):
#     client_df = df[df["partition_id"] == client_id]

#     for split in ["train", "valid", "test"]:
#         subset = client_df[client_df["split"] == split]
#         if subset.empty:
#             continue

#         plt.figure()
#         plt.bar(subset["class"], subset["count"])
#         plt.xlabel("Class")
#         plt.ylabel("Count")
#         plt.title(f"Client {client_id} – {split} class distribution")
#         plt.show()

# # ---------------------------------------------------------------------
# # 2. Global class distribution (all clients combined)
# # ---------------------------------------------------------------------

# global_dist = (
#     df.groupby("class")["count"]
#       .sum()
#       .reset_index()
# )

# plt.figure()
# plt.bar(global_dist["class"], global_dist["count"])
# plt.xlabel("Class")
# plt.ylabel("Total samples")
# plt.title("Global class distribution (all clients)")
# plt.show()

# # ---------------------------------------------------------------------
# # 3. Client × Class heatmap (train split)
# # ---------------------------------------------------------------------

# pivot = df[df["split"] == "train"].pivot_table(
#     index="partition_id",
#     columns="class",
#     values="count",
#     fill_value=0
# )

# plt.figure()
# plt.imshow(pivot, aspect="auto")
# plt.colorbar(label="Samples")
# plt.xlabel("Class")
# plt.ylabel("Client")
# plt.title("Train split: Client vs Class distribution")
# plt.show()

# # ---------------------------------------------------------------------
# # 4. Class coverage per client
# # ---------------------------------------------------------------------

# coverage = (
#     df[df["split"] == "train"]
#     .groupby("partition_id")["class"]
#     .nunique()
# )

# plt.figure()
# plt.bar(coverage.index, coverage.values)
# plt.xlabel("Client")
# plt.ylabel("Number of classes present")
# plt.title("Class coverage per client (train)")
# plt.show()

# # ---------------------------------------------------------------------
# # 5. Gini coefficient per client (label imbalance)
# # ---------------------------------------------------------------------

# def gini(x):
#     """
#     Gini coefficient: a scalar measure of inequality.
    
#     In this context, it quantifies how imbalanced a client's label
#     distribution is. A value close to 0 indicates a nearly uniform
#     distribution across classes (IID-like), while higher values
#     indicate stronger class imbalance (Non-IID behavior).
#     """
#     x = np.array(x, dtype=np.float64)
#     if np.all(x == 0):
#         return 0.0
#     x = np.sort(x)
#     n = len(x)
#     return (2 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1) / n

# gini_per_client = (
#     df[df["split"] == "train"]
#     .groupby("partition_id")["count"]
#     .apply(gini)
# )

# plt.figure()
# plt.bar(gini_per_client.index, gini_per_client.values)
# plt.xlabel("Client")
# plt.ylabel("Gini coefficient")
# plt.title("Label imbalance per client (train)")
# plt.show()

# # ---------------------------------------------------------------------
# # 6. Dataset size per experiment and split
# # ---------------------------------------------------------------------

# size_per_exp = (
#     df.groupby(["exp_id", "split"])["count"]
#       .sum()
#       .unstack(fill_value=0)
# )

# size_per_exp.plot(kind="bar")
# plt.ylabel("Number of samples")
# plt.title("Dataset size per experiment and split")
# plt.show()
