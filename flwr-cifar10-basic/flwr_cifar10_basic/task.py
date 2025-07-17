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
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Activation, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import he_normal
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
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
    # Model used Le-Net https://github.com/BIGBALLON/cifar-10-cnn/blob/master/README.md"
    weight_decay  = 0.00025
    
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(Conv2D(8, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(Dense(64, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
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
    # Shuffle test set
    test_set = test_set.shuffle(seed=42)
    print("-" * 80)
    print("Number of images Original Dataset TRAIN:", train_set.shape[0])
    print("Number of images Original Dataset TEST:", test_set.shape[0])

    # Original dataset: X_train (N, 32, 32, 3)
    # Let's generate 3 augmented datasets:
    # aug1 = np.array([augment_image(img) for img in train_set["img"]])
    # aug2 = np.array([augment_image(img) for img in train_set["img"]])

    # # Combine all:
    # X_train = np.concatenate((aug1, aug2), axis=0)

    # # Similarly for labels (assuming y_train shape (N,))
    # y_train = np.concatenate((train_set["label"], train_set["label"]), axis=0)
    
    # # Divide data on each node: 36000 images train, 12000 valid, 12000 test
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    X_train, X_valid, y_train, y_valid = train_test_split(train_set["img"], train_set["label"], test_size=0.2, random_state=42, stratify=train_set["label"])

    aug1 = np.array([augment_image(img) for img in X_train])
    aug2 = np.array([augment_image(img) for img in X_train])

    X_train = np.concatenate((aug1, aug2), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)

    # Images has to be float32 and labels int32
    # images["train"], labels["train"] = np.transpose(np.reshape(X_train / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1]), np.int32(y_train)
    # images["valid"], labels["valid"] = np.transpose(np.reshape(X_valid / 255.0, [-1, 3, 32, 32]), [0, 2, 3, 1]), np.int32(y_valid)

    images["train"], labels["train"] = (X_train / 255.0).astype("float32"), np.int32(y_train)
    images["valid"], labels["valid"] =(X_valid / 255.0).astype("float32"), np.int32(y_valid)
    
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    # print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    # print("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = np.float32((images["train"] - mean) / std)
    images["valid"] = np.float32((images["valid"] - mean) / std)
    
    images["test"] = (test_set["img"] / 255.0).astype("float32")
    images["test"], labels["test"] = np.float32((images["test"] - mean) / std), np.int32(test_set["label"])

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
