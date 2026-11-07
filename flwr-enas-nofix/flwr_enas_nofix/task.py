"""flwr-enas-nofix: A Flower / TensorFlow app."""

import os

# import keras
# from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
import numpy as np
import shutil
import sys

import time
from datetime import datetime
from absl import flags, app
import flwr_enas_nofix.src.framework as fw
import gc


from flwr_enas_nofix.src import utils
from flwr_enas_nofix.src.utils import Logger
from flwr_enas_nofix.src.utils import DEFINE_boolean
from flwr_enas_nofix.src.utils import DEFINE_integer
from flwr_enas_nofix.src.utils import DEFINE_string
from flwr_enas_nofix.src.utils import print_user_flags

from flwr_enas_nofix.src.cifar10.data_utils import read_data
from flwr_enas_nofix.src.cifar10.macro_controller import MacroController
from flwr_enas_nofix.src.cifar10.macro_child import MacroChild
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import json
# from src.cifar10.micro_controller import MicroController
# from src.cifar10.micro_child import MicroChild

import copy
from functools import lru_cache
import tomli
from pathlib import Path

import experiment_api as exp_api
import ablation_api as abla_api
import pandas as pd

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

fds = None  # Cache FederatedDataset

toml_path = Path(__file__).parent.parent / "pyproject.toml"
with toml_path.open("rb") as f:
    toml_config = tomli.load(f)
PROJECT_PATH = toml_config["tool"]["flwr"]["app"]["config"]["project-path"]
NON_IID = toml_config["tool"]["flwr"]["app"]["config"]["non_iid"]
ALPHA = toml_config["tool"]["flwr"]["app"]["config"]["alpha"]
DATASET = toml_config["tool"]["flwr"]["app"]["config"]["dataset"]
DATASET_NAME = toml_config["tool"]["flwr"]["app"]["config"]["dataset_name"]
LOCAL_EPOCHS = toml_config["tool"]["flwr"]["app"]["config"]["local-epochs"]
AGG_METHOD = toml_config["tool"]["flwr"]["app"]["config"]["agg_method"]
BATCH_SIZE = toml_config["tool"]["flwr"]["app"]["config"]["batch-size"]
MU = toml_config["tool"]["flwr"]["app"]["config"]["mu"]
NUM_ROUNDS = toml_config["tool"]["flwr"]["app"]["config"]["num-server-rounds"]
NUM_NODES = toml_config["tool"]["flwr"]["app"]["config"]["num-nodes"]
NUM_CLASSES = toml_config["tool"]["flwr"]["app"]["config"]["num-classes"]

BETA = toml_config["tool"]["flwr"]["app"]["config"]["beta"]
ABLATION = toml_config["tool"]["flwr"]["app"]["config"]["ablation"]

api = abla_api if ABLATION else exp_api

EARLY_STOPPING = False
EARLY_STOPPING_START_ROUND = 27
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0


def get_early_stopping_info_path():
    return api.get_controller_weights_dir() / "early_stoping_info.json"


def initialize_early_stopping_info():
    global EARLY_STOPPING
    EARLY_STOPPING = False

    path = get_early_stopping_info_path()
    payload = {"EARLY_STOPPING": False}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def read_early_stopping_flag():
    global EARLY_STOPPING
    path = get_early_stopping_info_path()

    if not path.exists():
        return EARLY_STOPPING

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        EARLY_STOPPING = bool(payload.get("EARLY_STOPPING", False))
    except Exception:
        pass

    return EARLY_STOPPING


def write_early_stopping_flag(value: bool):
    global EARLY_STOPPING
    EARLY_STOPPING = bool(value)

    path = get_early_stopping_info_path()
    payload = {"EARLY_STOPPING": EARLY_STOPPING}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def early_stopping_best_round(
    rounds: list[int],
    values: list[float],
    patience: int = EARLY_STOPPING_PATIENCE,
    min_delta: float = EARLY_STOPPING_MIN_DELTA,
    start_round: int = EARLY_STOPPING_START_ROUND,
) -> int:
    """
    Early stopping para una métrica a minimizar.
    Solo se evalúa desde start_round.
    Devuelve la mejor ronda.
    """
    if not rounds or not values:
        raise ValueError("rounds y values no pueden estar vacíos")
    if len(rounds) != len(values):
        raise ValueError("rounds y values deben tener la misma longitud")

    filtered = [(r, v) for r, v in zip(rounds, values) if r >= start_round]
    if not filtered:
        return rounds[-1]

    best_round = filtered[0][0]
    best_value = filtered[0][1]
    bad_rounds = 0

    for current_round, current_value in filtered[1:]:
        if current_value < (best_value - min_delta):
            best_value = current_value
            best_round = current_round
            bad_rounds = 0
        else:
            bad_rounds += 1

        if bad_rounds >= patience:
            return best_round

    return rounds[-1]


def write_experiment_txt():
    exp_dir = api.get_experiment_dir()
    path = exp_dir / "metadata_experiment.txt"

    if path.exists():
        return  # no lo sobrescribas
    
    initialize_early_stopping_info()

    with open(path, "w") as f:
        f.write(f"EXPERIMENT ID: {api.get_exp_id()}\n")
        f.write(f"STATUS: VALID\n") # valid | failed | discarded | debug
        f.write("REASON: N/A\n\n")
        f.write(f"DATE: {datetime.now()}\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"\tDATASET_NAME: {DATASET_NAME}\n")
        if NON_IID:
            f.write(f"\tDATA PARTITIONING: Non-IID with alpha={ALPHA}\n")
        
        else:
            f.write(f"\tDATA PARTITIONING: IID\n")
        f.write(f"\tAGG_METHOD: {AGG_METHOD}\n")
        f.write(f"\tNUM_ROUNDS: {NUM_ROUNDS}\n")
        f.write(f"\tNUM_NODES: {NUM_NODES}\n")
        if AGG_METHOD == "fedprox" or AGG_METHOD == "custom":
            f.write(f"\tMU: {MU}\n")
        if AGG_METHOD == "custom":
            f.write(f"\tBETA: {BETA}\n")
        f.write(f"\tBATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"\tLOCAL_EPOCHS: {LOCAL_EPOCHS}\n\n")
    



def smooth_curve(points, factor=0.8):
    points = np.array(points, dtype=np.float32)  # 👈 fuerza float
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# # Functions for controller
# def save_controller_weights(variables, filepath):
#     """
#     Save tf.Variables as a list of (name, numpy_array) tuples using pickle.
#     """
#     # Convertir variables a un diccionario {name: array}
#     weights_dict = {v.name.replace(":", "_"): v.numpy() for v in variables}

#     # Crear carpeta si no existe
#     filepath.parent.mkdir(parents=True, exist_ok=True)

#     # Guardar en formato .npz (comprimido)
#     np.savez_compressed(filepath, **weights_dict)
#     # print(f"[SAVE] Controller weights saved with names to: {filepath}")

# def load_controller_weights(filepath):
#     """
#     Load a list of (name, numpy_array) tuples and restore as tf.Variables with names.
#     """
#     # Abrir archivo .npz
#     with np.load(filepath) as data:
#         # Convertir cada array a tf.Variable
#         variables = [fw.Variable(data[key], name=key.replace("_", ":"), trainable=True)
#                      for key in data.files]

#     # Borrar archivo
#     try:
#         os.remove(filepath)
#     except FileNotFoundError:
#         pass

#     # Liberar memoria
#     gc.collect()

#     return variables


# Functions for controller
def save_controller_weights(variables, filepath):
    """
    Save tf.Variables as a list of (name, numpy_array) tuples using pickle.
    """
    weights = [(v.name, v.numpy()) for v in variables]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)
    # print(f"[SAVE] Controller weights saved with names to: {filepath}")

def load_controller_weights(filepath):
    """
    Load a list of (name, numpy_array) tuples and restore as tf.Variables with names.
    """
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)
    
    # Borrar archivo
    # try:
    #     os.remove(filepath)
    # except FileNotFoundError:
    #     pass

    # Liberar memoria
    # gc.collect()

    # print(f"WEIGHTS \n:{weights}")

    return weights
""" This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""

""" This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""


# Functions for child architectures
def weights_to_ndarrays(data_dict, keys):
    """
    Takes a dictionary with keys and array-like values,
    returns a tuple of:
    - A NumPy ndarray stacking all array values
    - A list of keys in the same order as the arrays
    """
    arrays = [data_dict[key] for key in keys]
    
    # try:
    #     stacked_array = np.stack(arrays)
    # except ValueError:
    #     # Fall back to np.array if shapes don't match
    #     stacked_array = np.array(arrays, dtype=object)
    return arrays

import numpy as np

def ndarray_to_weights(array_data, keys):
    """
    Takes a NumPy array (or list of arrays) and a list of keys,
    returns a dictionary mapping keys to their corresponding arrays.
    """
    if len(array_data) != len(keys):
        raise ValueError("Length of array_data and keys must be the same.")
    
    # Ensure array_data is a list of arrays
    array_data = np.asarray(array_data, dtype=object)

    return {key: fw.Variable(np.asarray(array_data[i]), name=key+":0",trainable=True) for i, key in enumerate(keys)}

global_label_distribution = None


def get_label_distribution(y, num_classes):
    y = np.asarray(y, dtype=np.int32)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    probs = counts / counts.sum()
    return probs


def total_variation_distance(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * np.sum(np.abs(p - q))


def init_global_label_distribution():
    """
    Calcula una sola vez la distribución global de etiquetas del split train completo.
    """
    global fds, global_label_distribution

    if global_label_distribution is not None:
        return global_label_distribution

    full_train = fds.load_split("train")
    full_train.set_format("numpy")

    num_classes = NUM_CLASSES

    if DATASET_NAME == "cifar100":
        y_global = np.asarray(full_train["fine_label"], dtype=np.int32)
    elif DATASET_NAME == "femnist":
        y_global = np.asarray(full_train["character"], dtype=np.int32)
    else:
        y_global = np.asarray(full_train["label"], dtype=np.int32)

    global_label_distribution = get_label_distribution(y_global, num_classes)
    return global_label_distribution


def compute_client_dis(y_local_raw):
    """
    Calcula:
      - distribución local del cliente
      - TV(local, global)
      - dis_i = 1 + TV
    usando las etiquetas originales del cliente.
    """
    num_classes = NUM_CLASSES
    p_global = init_global_label_distribution()
    p_local = get_label_distribution(y_local_raw, num_classes)
    tv_i = total_variation_distance(p_local, p_global)
    dis_i = 1.0 + tv_i
    return dis_i, tv_i, p_local, p_global

def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    images, labels = {}, {}

    def _to_32x32x3(x):
        x = np.asarray(x)
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim != 3:
            raise ValueError(f"Expected [N,H,W] or [N,H,W,1], got shape {x.shape}")
        if x.shape[1] == 28 and x.shape[2] == 28:
            x = np.pad(x, ((0, 0), (2, 2), (2, 2)), mode="constant")
        elif x.shape[1] != 32 or x.shape[2] != 32:
            raise ValueError(f"Expected spatial size 28x28 or 32x32, got {x.shape[1:3]}")
        x = x[..., np.newaxis]
        return np.repeat(x, 3, axis=-1)

    if DATASET_NAME == "cifar10":
        if fds is None:
            if NON_IID:
                # Use Dirichlet-based Non-IID partitioning with a minimum partition size to
                # prevent empty or degenerate client datasets that could cause unstable
                # training dynamics or inconsistent metric aggregation. 
                # min_partition_size=6500 is chosen to guarantee feasible Dirichlet splits
                # across 6 clients for CIFAR-10, while still preserving strong Non-IID
                # heterogeneity and ensuring each client has enough samples for stable
                # training and evaluation.
                partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=ALPHA, partition_by="label", min_partition_size=1500,
                                                self_balancing=True,
                                                seed=42,
                                                shuffle=True)
            else:
                partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset=DATASET,
                partitioners={"train": partitioner},
            )
            
        train_set = fds.load_partition(partition_id, "train")
        train_set.set_format("numpy")

        # Shuffle train set
        train_set = train_set.shuffle(seed=42)

        test_set = fds.load_split(split="test")
        test_set.set_format("numpy")
        # Shuffle test set
        test_set = test_set.shuffle(seed=42)
        print("-" * 80)
        print("Number of images Original Dataset TRAIN:", train_set.shape[0])
        print("Number of images Original Dataset TEST:", test_set.shape[0])
    
        # >>> TVD START
        y_local_raw = np.asarray(train_set["label"], dtype=np.int32)
        dis_i, tv_i, p_local, p_global = compute_client_dis(y_local_raw)
        # >>> TVD END
    
        # X_train = np.concatenate((train_set["img"], train_set["img"], train_set["img"]), axis=0) 
        # y_train = np.concatenate((train_set["label"], train_set["label"], train_set["label"]), axis=0)

        # X_train = train_set["img"]
        # y_train = train_set["label"]

        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # NEW DATA SPLIT 

        X = np.asarray(train_set["img"])
        y = np.asarray(train_set["label"])

        # Count samples per class
        classes, counts = np.unique(y, return_counts=True)

        print(f"Client {partition_id} class distribution:")
        print(dict(zip(classes.tolist(), counts.tolist())))

        # Identify classes that contain only one sample
        rare_classes = classes[counts < 2]

        # Keep rare-class samples exclusively in training
        rare_mask = np.isin(y, rare_classes)

        X_rare = X[rare_mask]
        y_rare = y[rare_mask]

        # Samples eligible for stratified splitting
        X_common = X[~rare_mask]
        y_common = y[~rare_mask]

        # Stratified split of classes having at least two samples
        common_classes = np.unique(y_common)
        n_valid = int(np.ceil(0.2 * len(y_common)))
        n_train = len(y_common) - n_valid

        if (
            len(common_classes) > 0
            and n_valid >= len(common_classes)
            and n_train >= len(common_classes)
        ):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_common,
                y_common,
                test_size=0.2,
                random_state=42,
                stratify=y_common
            )
        else:
            # Fallback if stratification is impossible
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=None
            )
            X_rare = X[:0]
            y_rare = y[:0]

        # Add rare-class samples to training
        if len(y_rare) > 0:
            X_train = np.concatenate((X_train, X_rare), axis=0)
            y_train = np.concatenate((y_train, y_rare), axis=0)

        # Duplicate after splitting
        X_train = np.concatenate((X_train, X_train, X_train), axis=0)
        y_train = np.concatenate((y_train, y_train, y_train), axis=0)

        images["train"], labels["train"] = np.float32(X_train) / 255.0, np.int32(y_train)
        images["valid"], labels["valid"] = np.float32(X_valid) / 255.0, np.int32(y_valid)
        
        mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
        std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

        # print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
        # print("std: {}".format(np.reshape(std * 255.0, [-1])))

        images["train"] = np.float32((images["train"] - mean) / std)
        images["valid"] = np.float32((images["valid"] - mean) / std)
        
        # print(test_set["img"])
        
        images["test"] = np.float32(test_set["img"]) / 255.0
        images["test"], labels["test"] = np.float32((images["test"] - mean) / std), np.int32(test_set["label"])
        
        
    elif DATASET_NAME == "cifar100":
        if fds is None:
            if NON_IID:
                # Use Dirichlet-based Non-IID partitioning with a minimum partition size to
                # prevent empty or degenerate client datasets that could cause unstable
                # training dynamics or inconsistent metric aggregation. 
                # min_partition_size=6500 is chosen to guarantee feasible Dirichlet splits
                # across 6 clients for CIFAR-10, while still preserving strong Non-IID
                # heterogeneity and ensuring each client has enough samples for stable
                # training and evaluation.
                partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=ALPHA, partition_by="fine_label", min_partition_size=1500,
                                                self_balancing=True,
                                                seed=42,
                                                shuffle=True)
            else:
                partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset=DATASET,
                partitioners={"train": partitioner},
            )
            
        train_set = fds.load_partition(partition_id, "train")
        train_set.set_format("numpy")

        # Shuffle train set
        train_set = train_set.shuffle(seed=42)

        test_set = fds.load_split(split="test")
        test_set.set_format("numpy")
        # Shuffle test set
        test_set = test_set.shuffle(seed=42)
        print("-" * 80)
        print("Number of images Original Dataset TRAIN:", train_set.shape[0])
        print("Number of images Original Dataset TEST:", test_set.shape[0])

        # >>> TVD START
        y_local_raw = np.asarray(train_set["fine_label"], dtype=np.int32)
        dis_i, tv_i, p_local, p_global = compute_client_dis(y_local_raw)
        # >>> TVD END

        # X_train = np.concatenate((train_set["img"], train_set["img"], train_set["img"]), axis=0)
        # y_train = np.concatenate((train_set['fine_label'], train_set['fine_label'], train_set['fine_label']), axis=0)
        
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # NEW DATA SPLIT 

        X = np.asarray(train_set["img"])
        y = np.asarray(train_set["label"])

        # Count samples per class
        classes, counts = np.unique(y, return_counts=True)

        print(f"Client {partition_id} class distribution:")
        print(dict(zip(classes.tolist(), counts.tolist())))

        # Identify classes that contain only one sample
        rare_classes = classes[counts < 2]

        # Keep rare-class samples exclusively in training
        rare_mask = np.isin(y, rare_classes)

        X_rare = X[rare_mask]
        y_rare = y[rare_mask]

        # Samples eligible for stratified splitting
        X_common = X[~rare_mask]
        y_common = y[~rare_mask]

        # Stratified split of classes having at least two samples
        common_classes = np.unique(y_common)
        n_valid = int(np.ceil(0.2 * len(y_common)))
        n_train = len(y_common) - n_valid

        if (
            len(common_classes) > 0
            and n_valid >= len(common_classes)
            and n_train >= len(common_classes)
        ):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_common,
                y_common,
                test_size=0.2,
                random_state=42,
                stratify=y_common
            )
        else:
            # Fallback if stratification is impossible
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=None
            )
            X_rare = X[:0]
            y_rare = y[:0]

        # Add rare-class samples to training
        if len(y_rare) > 0:
            X_train = np.concatenate((X_train, X_rare), axis=0)
            y_train = np.concatenate((y_train, y_rare), axis=0)

        # Duplicate after splitting
        X_train = np.concatenate((X_train, X_train, X_train), axis=0)
        y_train = np.concatenate((y_train, y_train, y_train), axis=0)

        images["train"], labels["train"] = np.float32(X_train) / 255.0, np.int32(y_train)
        images["valid"], labels["valid"] = np.float32(X_valid) / 255.0, np.int32(y_valid)
        
        mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
        std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

        # print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
        # print("std: {}".format(np.reshape(std * 255.0, [-1])))

        images["train"] = np.float32((images["train"] - mean) / std)
        images["valid"] = np.float32((images["valid"] - mean) / std)
        
        # print(test_set["img"])
        
        images["test"] = np.float32(test_set["img"]) / 255.0
        images["test"], labels["test"] = np.float32((images["test"] - mean) / std), np.int32(test_set["fine_label"])

    elif DATASET_NAME == "mnist":
        if fds is None:
            if NON_IID:
                # Use Dirichlet-based Non-IID partitioning with a minimum partition size to
                # prevent empty or degenerate client datasets that could cause unstable
                # training dynamics or inconsistent metric aggregation. 
                # min_partition_size=6500 is chosen to guarantee feasible Dirichlet splits
                # across 6 clients for CIFAR-10, while still preserving strong Non-IID
                # heterogeneity and ensuring each client has enough samples for stable
                # training and evaluation.
                partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=ALPHA, partition_by="label", min_partition_size=1500,
                                                self_balancing=True,
                                                seed=42,
                                                shuffle=True)
            else:
                partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset=DATASET,
                partitioners={"train": partitioner},
            )
            
        train_set = fds.load_partition(partition_id, "train")
        train_set.set_format("numpy")

        # Shuffle train set
        train_set = train_set.shuffle(seed=42)

        test_set = fds.load_split(split="test")
        test_set.set_format("numpy")
        # Shuffle test set
        test_set = test_set.shuffle(seed=42)
        print("-" * 80)
        print("Number of images Original Dataset TRAIN:", train_set.shape[0])
        print("Number of images Original Dataset TEST:", test_set.shape[0])

        # >>> TVD START
        y_local_raw = np.asarray(train_set["label"], dtype=np.int32)
        dis_i, tv_i, p_local, p_global = compute_client_dis(y_local_raw)
        # >>> TVD END

        # X_train = np.concatenate((train_set["image"], train_set["image"], train_set["image"]), axis=0)
        # X_train = _to_32x32x3(X_train)
        # y_train = np.concatenate((train_set['label'], train_set['label'], train_set['label']), axis=0)
        
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # NEW DATA SPLIT 

        X = np.asarray(train_set["img"])
        y = np.asarray(train_set["label"])

        # Count samples per class
        classes, counts = np.unique(y, return_counts=True)

        print(f"Client {partition_id} class distribution:")
        print(dict(zip(classes.tolist(), counts.tolist())))

        # Identify classes that contain only one sample
        rare_classes = classes[counts < 2]

        # Keep rare-class samples exclusively in training
        rare_mask = np.isin(y, rare_classes)

        X_rare = X[rare_mask]
        y_rare = y[rare_mask]

        # Samples eligible for stratified splitting
        X_common = X[~rare_mask]
        y_common = y[~rare_mask]

        # Stratified split of classes having at least two samples
        common_classes = np.unique(y_common)
        n_valid = int(np.ceil(0.2 * len(y_common)))
        n_train = len(y_common) - n_valid

        if (
            len(common_classes) > 0
            and n_valid >= len(common_classes)
            and n_train >= len(common_classes)
        ):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_common,
                y_common,
                test_size=0.2,
                random_state=42,
                stratify=y_common
            )
        else:
            # Fallback if stratification is impossible
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=None
            )
            X_rare = X[:0]
            y_rare = y[:0]

        # Add rare-class samples to training
        if len(y_rare) > 0:
            X_train = np.concatenate((X_train, X_rare), axis=0)
            y_train = np.concatenate((y_train, y_rare), axis=0)

        # Duplicate after splitting
        X_train = np.concatenate((X_train, X_train, X_train), axis=0)
        y_train = np.concatenate((y_train, y_train, y_train), axis=0)

        images["train"], labels["train"] = np.float32(X_train) / 255.0, np.int32(y_train)
        images["valid"], labels["valid"] = np.float32(X_valid) / 255.0, np.int32(y_valid)
        
        mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
        std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

        # print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
        # print("std: {}".format(np.reshape(std * 255.0, [-1])))

        images["train"] = np.float32((images["train"] - mean) / std)
        images["valid"] = np.float32((images["valid"] - mean) / std)
        
        # print(test_set["img"])
        
        images["test"] = np.float32(_to_32x32x3(test_set["image"])) / 255.0
        images["test"], labels["test"] = np.float32((images["test"] - mean) / std), np.int32(test_set["label"])

    elif DATASET_NAME == "femnist":
        if fds is None:
            if NON_IID:
                # Use Dirichlet-based Non-IID partitioning with a minimum partition size to
                # prevent empty or degenerate client datasets that could cause unstable
                # training dynamics or inconsistent metric aggregation. 
                # min_partition_size=6500 is chosen to guarantee feasible Dirichlet splits
                # across 6 clients for CIFAR-10, while still preserving strong Non-IID
                # heterogeneity and ensuring each client has enough samples for stable
                # training and evaluation.
                partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=ALPHA, partition_by="character", min_partition_size=1500,
                                                self_balancing=True,
                                                seed=42,
                                                shuffle=True)
            else:
                partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset=DATASET,
                partitioners={"train": partitioner},
            )
            
        train_set = fds.load_partition(partition_id, "train")
        train_set.set_format("numpy")

        # Shuffle train set
        train_set = train_set.shuffle(seed=42)

        print("-" * 80)
        print("Number of images Original Dataset TRAIN:", train_set.shape[0])

        # >>> TVD START
        y_local_raw = np.asarray(train_set["character"], dtype=np.int32)
        dis_i, tv_i, p_local, p_global = compute_client_dis(y_local_raw)
        # >>> TVD END

        X_train = np.array(train_set["image"])
        X_train = _to_32x32x3(X_train)
        y_train = np.array(train_set['character'])
        
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/7, random_state=42, stratify=y_train)

        images["train"], labels["train"] = np.float32(X_train) / 255.0, np.int32(y_train)
        images["valid"], labels["valid"] = np.float32(X_valid) / 255.0, np.int32(y_valid)
        
        mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
        std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

        # print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
        # print("std: {}".format(np.reshape(std * 255.0, [-1])))

        images["train"] = np.float32((images["train"] - mean) / std)
        images["valid"] = np.float32((images["valid"] - mean) / std)
        
        # print(test_set["img"])
        
        images["test"] = X_test / 255.0
        images["test"], labels["test"] = np.float32((images["test"] - mean) / std), np.int32(y_test)

    print("Number of images in dataset (Train):", images["train"].shape[0])
    print("Number of images in dataset (Valid):", images["valid"].shape[0])
    print("Number of images in dataset (Test):", images["test"].shape[0])
    
    rows = []

    print("-" * 80)
    print("Label distribution:")

    for split in ["train", "valid", "test"]:
        label_counts = Counter(labels[split])
        print(f"{split.upper()} set class distribution:")
        for label in sorted(label_counts):
            count = label_counts[label]
            print(f"  Class {label}: {count} samples")
            rows.append({
                "exp_id": api.get_exp_id(),
                "partition_id": partition_id,
                "dataset_name": DATASET_NAME,
                "split": split,
                "class": int(label),
                "count": int(count),
            })
        print("-" * 40)

    print("-" * 80)

    df = pd.DataFrame(rows)

    df.to_parquet(
        api.get_label_distributions_dir() / f"client_{partition_id}.parquet",
        engine="pyarrow",
        index=False,
    )

    # To manage this in the future: df.groupby(["exp_id", "dataset_name", "split", "class"])["count"].sum()
    
    return {"images": images, "labels": labels, "dis_i": dis_i}
    

        

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "data/cifar10", "")
DEFINE_string("output_dir", "outputs", "")

DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 1,
               "train the controller after this number of epochs") #This values has to be multiple of eval_every_epochs
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")
DEFINE_integer("num_epochs", LOCAL_EPOCHS, "How many epochs to train")
DEFINE_boolean("child_use_aux_heads", True, "")
    
class Trainer:
    def __init__(self, dataset=None, arch=None, transfer=False, actual_round=None, partition_id=None, learning_rate=0.05):
        self.dataset = dataset
        self.arch = arch
        self.transfer = transfer
        self.actual_round = actual_round
        self.partition_id = partition_id
        self.learning_rate = learning_rate
        self.ops = {}
        
    def get_ops(self):
        """
        Args:
            images: dict with keys {"train", "valid", "test"}.
            labels: dict with keys {"train", "valid", "test"}.
        """
        FLAGS = flags.FLAGS
        assert FLAGS.search_for is not None, "Please specify --search_for"

        # Controller and Child class selection
        ControllerClass = MacroController
        ChildClass = MacroChild

        # Initialize child model

        if AGG_METHOD == "fedavg":

            child_model = ChildClass(
                self.dataset["images"],
                self.dataset["labels"],
                clip_mode="norm",
                optim_algo="sgd",
                child_weights=self.arch["child_weights"],
                transfer=self.transfer,
                mu = 0.0,
                learning_rate=self.learning_rate,
                num_clases=NUM_CLASSES
            )
        
        else:
            child_model = ChildClass(
                self.dataset["images"],
                self.dataset["labels"],
                clip_mode="norm",
                optim_algo="sgd",
                child_weights=self.arch["child_weights"],
                transfer=self.transfer,
                mu = MU,
                learning_rate=self.learning_rate,
                num_clases=NUM_CLASSES
            )
        
        controller_ops = None
        dataset_valid_shuffle = None
        
        # Initialize controller model if no fixed architecture is provided
        if FLAGS.child_fixed_arc is None:
            controller_model = ControllerClass(
                lstm_size=64,
                lstm_num_layers=1,
                lstm_keep_prob=1.0,
                lr_dec_start=0,
                lr_dec_every=1000000,  # never decrease learning rate
                optim_algo="adam",
                controller_weights=self.arch["controller_trainable_variables"],
                transfer=self.transfer
            )
            
            # Connect controller with child model
            child_train_op, child_lr, child_optimizer = child_model.connect_controller(controller_model)
            dataset_valid_shuffle = child_model.ValidationRLShuffle(
                child_model, False
            )(child_model.images['valid_original'], child_model.labels['valid_original'])

            # Build controller trainer
            controller_train_op, controller_lr, controller_optimizer = controller_model.build_trainer(
                child_model, child_model.ValidationRL()
            )

            # Collect controller operations
            controller_ops = {
                "train_step": controller_model.train_step,
                "generate_sample_arc": controller_model.generate_sample_arc,
                'loss': controller_model.loss,
                "train_op": controller_train_op,
                "lr": controller_lr,
                'trainable_variables': controller_model.trainable_variables(),
                "valid_acc": controller_model.valid_acc,
                "optimizer": controller_optimizer,
                "baseline": controller_model.baseline,
                "entropy": lambda: controller_model.current_entropy,
            }

        else:
            # Handle case where architecture is fixed
            assert not FLAGS.controller_training, (
                "--child_fixed_arc is given, cannot train controller"
            )
            child_train_op, child_lr, child_optimizer = child_model.connect_controller(None)

        # Clean up to free memory

        self.ops = {
            "child": {
                'generate_train_losses': child_model.generate_train_losses,
                'test_model': child_model.test_model,
                'validation_model': child_model.valid_model,
                'validation_rl_model': child_model.valid_rl_model,
                'global_step': child_model.global_step,
                'dataset': child_model.dataset,
                'dataset_test': child_model.dataset_test,
                'dataset_valid_shuffle': dataset_valid_shuffle,
                "loss": child_model.loss,
                "train_loss": child_model.train_loss,
                "train_op": child_train_op,
                "lr": child_lr,
                'trainable_variables': child_model.trainable_variables(),
                "optimizer": child_optimizer,
                "num_train_batches": child_model.num_train_batches,
                "weights": child_model.weights.weight_map,
            },
            "controller": controller_ops,
            "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
            "eval_func": child_model.eval_once,
            "num_train_batches": child_model.num_train_batches,
        }
        # Ensure previous sessions and variables are cleared to free memory
        child_model, controller_ops, dataset_valid_shuffle, child_valid_rl_model, controller_model = [None] * 5

    @fw.function(autograph=False)
    def child_train_op(self, images, labels):
        with fw.GradientTape() as tape:
            child_train_logits, child_loss, child_train_loss, child_train_acc = self.ops['child']['generate_train_losses'](images, labels)
            child_grad_norm, child_grad_norm_list, _ = self.ops['child']['train_op'](child_loss, self.ops['child']['trainable_variables'], tape)
        return child_train_logits, child_loss, child_train_acc, child_grad_norm
    
    @fw.function(autograph=False)
    def controller_train_op(self, child_train_logits, labels):
        with fw.GradientTape() as tape:
            self.ops["controller"]["generate_sample_arc"]()
            controller_loss = self.ops['controller']['loss'](child_train_logits, labels)
            gn, gn_list, _ = self.ops['controller']['train_op'](controller_loss, self.ops['controller']['trainable_variables'], tape)
            return controller_loss, gn

    def train(self):
        # tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()
        # gc.collect()
        fw.clear_session()
        FLAGS = flags.FLAGS
        # images, labels = data["images"], data["labels"]
        # if self.actual_round == 1:
        #     print("*" * 80)
        #     print("CONFIGURATION:")
        #     print()
        #     print(f"EXPERIMENT ID: {get_exp_id()}\n")
        #     print(f"STATUS: VALID\n") # valid | failed | discarded | debug
        #     print("REASON: N/A\n\n")
        #     print(f"DATE: {datetime.now()}\n\n")
        #     print(f"\tDATASET_NAME: {DATASET_NAME}\n")
        #     if NON_IID:
        #         print(f"\tDATA PARTITIONING: Non-IID with alpha={ALPHA}\n")
            
        #     else:
        #         print(f"\tDATA PARTITIONING: IID\n")
        #     print(f"\tAGG_METHOD: {AGG_METHOD}\n")
        #     print(f"\tNUM_ROUNDS: {NUM_ROUNDS}\n")
        #     print(f"\tNUM_NODES: {NUM_NODES}\n")
        #     if AGG_METHOD != "fedavg":
        #         print(f"\tMU: {MU}\n")
        #     print(f"\tBATCH_SIZE: {BATCH_SIZE}\n")
        #     print(f"\tLOCAL_EPOCHS: {LOCAL_EPOCHS}\n\n")
        #     print("*" * 80)
        
        self.get_ops()
        batch_iterator = None

        print("-" * 80)
        print("Starting session")
        start_time = datetime.now()

        child_train_accuracies = []
        child_train_accs = []
        child_train_lss = []
        child_valid_accuracies = []
        child_train_losses = []
        child_valid_losses = []

        controller_accs = []
        controller_lss = []
        controller_accuracies = []
        controller_losses = []
        while True:
            if read_early_stopping_flag():
                print(f"EARLY_STOPPING = True. Se devuelven los últimos parámetros agregados.")
                child_valid_acc, child_valid_loss, child_valid_f1_score = self.ops["eval_func"]("valid", self.ops['child']['validation_model'], verbose=True)
                child_test_acc, child_test_loss, child_test_f1_score = self.ops["eval_func"]("test", self.ops['child']['test_model'], verbose=True)
                break

            if batch_iterator is None:
                batch_iterator = self.ops['child']['dataset'].as_numpy_iterator()
            try:
                images, labels = next(batch_iterator)
            except StopIteration:
                batch_iterator = self.ops['child']['dataset'].as_numpy_iterator()
                images, labels = next(batch_iterator)
            
            child_train_logits, child_loss, child_train_acc, child_grad_norm = self.child_train_op(images, labels)
            child_lr = self.ops['child']['lr']()
            global_step = self.ops["child"]["global_step"].value()
            child_train_accs.append(float(child_train_acc/FLAGS.batch_size))
            child_train_lss.append(float(child_loss))
            if FLAGS.child_sync_replicas:
                actual_step = global_step * FLAGS.child_num_aggregate
            else:
                actual_step = global_step

            epoch = actual_step // self.ops["num_train_batches"]

            if global_step % FLAGS.log_every == 0:
                log_string = f"epoch={epoch:<6d}\tch_step={global_step:<6d}\tloss={child_loss:.4f}\t"
                log_string += f"lr={child_lr:.4f}\t|g|={child_grad_norm:.4f}\ttr_acc={(child_train_acc/FLAGS.batch_size):4f}\t"
                # log_string += f"Time: {curr_time - start_time}"
                print(log_string)

            if actual_step % self.ops["eval_every"] == 0:
                # images_batch, labels_batch = next(self.ops['child']['dataset_valid_shuffle'].as_numpy_iterator())
                # child_valid_rl_logits = self.ops['child']['validation_rl_model'](images_batch)
                
                # test_images_batch, test_labels_batch = next(self.ops['child']['dataset_test'].as_numpy_iterator())
                # print(f"TEST IMAGES BATCH: {test_images_batch.shape[0]}")
                # child_test_logits = self.ops['child']['test_model'](test_images_batch)
                
                print("-" * 80)
                num_archs = 2
                print(f"Here are {num_archs} architectures")
                for _ in range(num_archs):
                    arc = self.ops["controller"]["generate_sample_arc"]()
                    # print(arc)
                    images_batch, labels_batch = next(self.ops['child']['dataset_valid_shuffle'].as_numpy_iterator())
                    child_valid_rl_logits = self.ops['child']['validation_rl_model'](images_batch)
                    acc = self.ops["controller"]["valid_acc"](child_valid_rl_logits, labels_batch)
                    if FLAGS.search_for == "micro":
                        normal_arc, reduce_arc = arc
                        print(np.reshape(normal_arc, [-1]))
                        print(np.reshape(reduce_arc, [-1]))
                    else:
                        start = 0
                        for layer_id in range(FLAGS.child_num_layers):
                            if FLAGS.controller_search_whole_channels:
                                end = start + 1 + layer_id
                            else:
                                end = start + 2 * FLAGS.child_num_branches + layer_id
                            print(np.reshape(arc[start: end], [-1]))
                            start = end
                    print(f"val_acc={acc:<6.4f}")
                    print("-" * 80)
                
                print(f"Epoch {epoch}: Eval")
                # child_valid_acc = self.ops["eval_func"]("valid", child_valid_rl_logits, labels_batch,verbose=False)
                # child_test_acc = self.ops["eval_func"]("test", child_test_logits, test_labels_batch,verbose=False)
                print(f"train_accuracy: {sum(child_train_accs)/len(child_train_accs)}")
                print(f"train_loss: {sum(child_train_lss)/len(child_train_lss)}")
                child_valid_acc, child_valid_loss, child_valid_f1_score = self.ops["eval_func"]("valid", self.ops['child']['validation_model'], verbose=True)
                
                child_train_accuracies.append(sum(child_train_accs)/len(child_train_accs))
                child_train_losses.append(sum(child_train_lss)/len(child_train_lss))
                child_valid_accuracies.append(float(child_valid_acc))
                child_valid_losses.append(float(child_valid_loss))
                print("-" * 80)
                if FLAGS.controller_training and epoch % FLAGS.controller_train_every == 0:
                    print(f"Epoch {epoch}: Training controller")
                    images_batch, labels_batch = next(self.ops['child']['dataset_valid_shuffle'].as_numpy_iterator())
                    child_valid_rl_logits = self.ops['child']['validation_rl_model'](images_batch)
                    for ct_step in range(FLAGS.controller_train_steps * FLAGS.controller_num_aggregate):
                        # images_batch, labels_batch = next(self.ops['child']['dataset_valid_shuffle'].as_numpy_iterator())
                        # child_valid_rl_logits = self.ops['child']['validation_rl_model'](images_batch)
                        controller_valid_acc = self.ops['controller']['valid_acc'](child_valid_rl_logits, labels_batch)
                        controller_loss, gn = self.controller_train_op(child_valid_rl_logits, labels_batch)
                        controller_entropy = 0
                        lr = self.ops["controller"]["lr"]()
                        bl = self.ops["controller"]["baseline"]

                        controller_accs.append(float(controller_valid_acc))
                        controller_lss.append(float(controller_loss))
                        if ct_step % FLAGS.log_every == 0:
                            curr_time = datetime.now()
                            log_string = f"ctrl_step={self.ops['controller']['train_step'].value():<6d}\tloss={controller_loss:.4f}\t"
                            log_string += f"ent={controller_entropy:.4f}\tlr={lr:.4f}\t|g|={gn:.4f}\tacc={controller_valid_acc:.4f}\t"
                            log_string += f"bl={bl.value():4f}\t"
                            # log_string + = f"Time: {curr_time - start_time}"
                            print(log_string)
                    controller_accuracies.append(sum(controller_accs)/len(controller_accs))
                    controller_losses.append(sum(controller_lss)/len(controller_lss))
                    print("-" * 80)
                
                curr_time = datetime.now()
                print(f"Time Running: {curr_time - start_time}")
                print("-" * 80)

            if epoch >= FLAGS.num_epochs:
                print("FINAL EVALUATION: ")
                child_valid_acc, child_valid_loss, child_valid_f1_score = self.ops["eval_func"]("valid", self.ops['child']['validation_model'], verbose=True)
                child_test_acc, child_test_loss, child_test_f1_score = self.ops["eval_func"]("test", self.ops['child']['test_model'], verbose=True)
                num_archs = 10
                print(f"Here are FINAL {num_archs} architectures")
                best_arc = None
                best_acc = -1
                for num in range(num_archs):
                    arc = self.ops["controller"]["generate_sample_arc"]()
                    print(f"ARC {num}")
                    print(arc)
                    images_batch, labels_batch = next(self.ops['child']['dataset_valid_shuffle'].as_numpy_iterator())
                    child_valid_rl_logits = self.ops['child']['validation_rl_model'](images_batch)
                    acc = self.ops["controller"]["valid_acc"](child_valid_rl_logits, labels_batch)
                    print(f"val_acc={acc:<6.4f}")
                    if acc >= best_acc:
                        best_acc = acc
                        best_arc = arc
                    print("+" * 20)
                break
        
        print("-" * 80)
        curr_time = datetime.now()
        print(f"Time Running: {curr_time - start_time}")
        print("-" * 80)
        return {
                "child_valid_acc": float(child_valid_acc),
                "child_valid_loss": float(child_valid_loss),
                "child_valid_f1_score": float(child_valid_f1_score),
                "child_test_acc": float(child_test_acc),
                "child_test_loss": float(child_test_loss),
                "child_test_f1_score": float(child_test_f1_score),
                "child_weights": self.ops["child"]["weights"],
                "child_trainable_variables": self.ops['child']['trainable_variables'],
                "controller_trainable_variables": self.ops["controller"]["trainable_variables"],
                "final_arc": best_arc
                }
