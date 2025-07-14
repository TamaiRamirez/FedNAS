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
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Activation, BatchNormalization
from keras import optimizers
from keras.initializers import he_normal
from sklearn.model_selection import train_test_split

import tomli
from pathlib import Path
from collections import Counter

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*7)])
  except RuntimeError as e:
    print(e)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


toml_path = Path(__file__).parent.parent / "pyproject.toml"
with toml_path.open("rb") as f:
    toml_config = tomli.load(f)
PROJECT_PATH = toml_config["tool"]["flwr"]["app"]["config"]["project-path"]


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    # Model used VGG 19 https://github.com/BIGBALLON/cifar-10-cnn/blob/master/README.md"
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Freeze the base model
    inputs = Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=he_normal(), name='fc_cifa10')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=he_normal(), name='fc_c2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, use_bias = True, kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=he_normal(), name='predictions')(x)
    x = BatchNormalization()(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs, outputs)

    sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model


def augment_image(img, crop_size=32, pad_size=4, flip_prob=0.5):
    """
    img: input image as a NumPy array of shape (H, W, C) with H=W=32
    Returns: augmented image of shape (32, 32, C)
    """

    # 1. Center pad to 40x40
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')

    # 2. Random crop
    h, w = padded_img.shape[:2]
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    cropped_img = padded_img[top:top + crop_size, left:left + crop_size, :]

    # 3. Random horizontal flip
    if np.random.rand() < flip_prob:
        cropped_img = np.fliplr(cropped_img)

    return cropped_img


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

    train_set = fds.load_partition(partition_id, "train")
    train_set.set_format("numpy")

    # Shuffle train set
    train_set = train_set.shuffle(seed=42)

    test_set = fds.load_split(split="test")
    test_set.set_format("numpy")

    print("-" * 80)
    print("Number of images Original Dataset TRAIN:", train_set.shape[0])
    print("Number of images Original Dataset TEST:", test_set.shape[0])

    # Original dataset: X_train (N, 32, 32, 3)
    # Let's generate 3 augmented datasets:
    aug1 = np.array([augment_image(img) for img in train_set["img"]])
    aug2 = np.array([augment_image(img) for img in train_set["img"]])

    # Combine all:
    X_train = np.concatenate((aug1, aug2), axis=0)

    # Similarly for labels (assuming y_train shape (N,))
    y_train = np.concatenate((train_set["label"], train_set["label"]), axis=0)
    
    # Divide data on each node: 36000 images train, 12000 valid, 12000 test
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Images has to be float32 and labels int32
    images["train"], labels["train"] = np.transpose(np.reshape(X_train / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1]), y_train
    images["valid"], labels["valid"] = np.transpose(np.reshape(X_valid / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1]), y_valid
    
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    # print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    # print("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std
    images["valid"] = (images["valid"] - mean) / std
    
    images["test"] = np.transpose(np.reshape(test_set["img"] / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1])
    images["test"], labels["test"] = (images["test"] - mean) / std, test_set["label"]

    print("Number of images in dataset (Train):", images["train"].shape[0])
    print("Number of images in dataset (Valid):", images["valid"].shape[0])
    print("Number of images in dataset (Test):", images["test"].shape[0])
    

    print("-" * 80)
    print("Label distribution:")
    for split in ["train", "valid", "test"]:
        label_counts = Counter(labels[split])
        print(f"{split.upper()} set class distribution:")
        for cls in sorted(label_counts):
            print(f"  Class {cls}: {label_counts[cls]} samples")
        print("-" * 40)
    print("-" * 80)
    return images, labels
