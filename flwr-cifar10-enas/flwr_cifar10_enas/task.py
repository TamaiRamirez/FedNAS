"""flwr-cifar10-enas: A Flower / TensorFlow app."""

import os

import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import numpy as np
import shutil
import sys

import time
from datetime import datetime
from absl import flags, app
import flwr_cifar10_enas.src.framework as fw
from flwr_cifar10_enas.src import utils
from flwr_cifar10_enas.src.utils import Logger
from flwr_cifar10_enas.src.utils import DEFINE_boolean
from flwr_cifar10_enas.src.utils import DEFINE_integer
from flwr_cifar10_enas.src.utils import DEFINE_string
from flwr_cifar10_enas.src.utils import print_user_flags

from flwr_cifar10_enas.src.cifar10.data_utils import read_data
from flwr_cifar10_enas.src.cifar10.macro_controller import MacroController
from flwr_cifar10_enas.src.cifar10.macro_child import MacroChild
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

# from src.cifar10.micro_controller import MicroController
# from src.cifar10.micro_child import MicroChild

import copy
from functools import lru_cache
import tomli
from pathlib import Path

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

fds = None  # Cache FederatedDataset

toml_path = Path(__file__).parent.parent / "pyproject.toml"
with toml_path.open("rb") as f:
    toml_config = tomli.load(f)
PROJECT_PATH = toml_config["tool"]["flwr"]["app"]["config"]["project-path"]

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

    return [tf.Variable(array, name=name, trainable=True) for name, array in weights]
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
    
    try:
        stacked_array = np.stack(arrays)
    except ValueError:
        # Fall back to np.array if shapes don't match
        stacked_array = np.array(arrays, dtype=object)

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


    X_train, X_valid, y_train, y_valid = train_test_split(train_set["img"], train_set["label"], test_size=0.2, random_state=42, stratify=train_set["label"])

    X_train = np.concatenate((X_train, X_train), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)

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
    # print("Label distribution:")
    # for split in ["train", "valid", "test"]:
    #     label_counts = Counter(labels[split])
    #     print(f"{split.upper()} set class distribution:")
    #     for cls in sorted(label_counts):
    #         print(f"  Class {cls}: {label_counts[cls]} samples")
    #     print("-" * 40)
    # print("-" * 80)
    return {"images": images, "labels": labels}


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "data/cifar10", "")
DEFINE_string("output_dir", "outputs", "")

DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 1,
               "train the controller after this number of epochs") #This values has to be multiple of eval_every_epochs
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")
DEFINE_integer("num_epochs", 20, "How many epochs to train")
DEFINE_boolean("child_use_aux_heads", True, "")

from memory_profiler import profile
    
class Trainer:
    def __init__(self, dataset=None, arch=None, transfer=False, actual_round=None, partition_id=None):
        self.dataset = dataset
        self.arch = arch
        self.transfer = transfer
        self.actual_round = actual_round
        self.partition_id = partition_id
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
        child_model = ChildClass(
            self.dataset["images"],
            self.dataset["labels"],
            clip_mode="norm",
            optim_algo="momentum",
            child_weights=self.arch["child_weights"],
            transfer=self.transfer
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
        FLAGS = flags.FLAGS
        # images, labels = data["images"], data["labels"]
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
                child_valid_acc, child_valid_loss = self.ops["eval_func"]("valid", self.ops['child']['validation_model'], verbose=True)
                
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
                print("Test Accuracy: ")
                child_test_acc, child_test_loss = self.ops["eval_func"]("test", self.ops['child']['test_model'], verbose=True)

                # Directorio de guardado
                plot_dir = f"{PROJECT_PATH}/plots/client_{self.partition_id}/round_{self.actual_round}"
                os.makedirs(plot_dir, exist_ok=True)
                
                # Gráfico de precisión (accuracy)
                plt.figure()
                plt.plot(smooth_curve(child_train_accuracies), label="Child Train Accuracy")
                plt.plot(smooth_curve(child_valid_accuracies), label="Child Validation Accuracy")
                plt.plot(smooth_curve(child_train_losses), label="Child Train Loss")
                plt.plot(smooth_curve(child_valid_losses), label="Child Validation Loss")
                # Línea horizontal punteada en y=1
                plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5)
                plt.title(f"Client {self.partition_id} - Child Metrics (Round {self.actual_round})")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                # plt.ylim(0, 2)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/child_metrics.png")
                plt.close()
                
                # Gráfico de precisión (accuracy)
                plt.figure()
                plt.plot(smooth_curve(controller_accuracies), label="Controller Accuracy")
                plt.plot(smooth_curve(controller_losses), label="Controller Loss")
                # Línea horizontal punteada en y=1
                plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5)
                plt.title(f"Client {self.partition_id} - Controller Metrics (Round {self.actual_round})")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                # plt.ylim(0, 2)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/controller_metrics.png")
                plt.close()
                
                break

        return {
                "child_valid_acc": float(child_valid_acc),
                "child_valid_loss": float(child_valid_loss),
                "child_test_acc": float(child_test_acc),
                "child_test_loss": float(child_test_loss),
                "child_weights": self.ops["child"]["weights"],
                "controller_trainable_variables": self.ops["controller"]["trainable_variables"]
                }
