"""flwr-cifar10-basic: A Flower / TensorFlow app."""

import os

import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    # Model used VGG 19 https://github.com/BIGBALLON/cifar-10-cnn/blob/master/README.md"
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Freeze the base model
    inputs = Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
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

    # Divide data on each node: 40000 images train, 10000 valid, 10000 test
    partition = partition.train_test_split(test_size=0.17)
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
