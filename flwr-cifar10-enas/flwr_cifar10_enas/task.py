"""flwr-cifar10-enas: A Flower / TensorFlow app."""

import os

import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import numpy as np


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    images, labels = {}, {}
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    partition_train=partition["train"].train_test_split(test_size=0.2)
    images["train"], labels["train"] = np.transpose(np.reshape(partition_train["train"]["img"] / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1]), partition_train["train"]["label"]
    images["valid"], labels["valid"] = np.transpose(np.reshape(partition_train["test"]["img"] / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1]), partition_train["test"]["label"]
    images["test"], labels["test"] = np.transpose(np.reshape(partition["test"]["img"] / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1]), partition["test"]["label"]
    
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std
    images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std


    return images, labels
