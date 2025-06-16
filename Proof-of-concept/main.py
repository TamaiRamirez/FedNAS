import os
import pickle as pickle
import shutil
import sys
import time
from datetime import datetime
import numpy as np
from absl import flags, app
import src.framework as fw
import pickle
from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.cifar10.data_utils import read_data
from src.cifar10.macro_controller import MacroController
from src.cifar10.macro_child import MacroChild

# from src.cifar10.micro_controller import MicroController
# from src.cifar10.micro_child import MicroChild

import copy
from functools import lru_cache
# import psutil
# import gc

# gc.enable()
# inner psutil function
# def process_memory():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss

# # decorator function
# def profile(func):
#     def wrapper(*args, **kwargs):

#         mem_before = process_memory() / (1024*1024*1024)
#         result = func(*args, **kwargs)
#         mem_after = process_memory() / (1024*1024*1024)
#         print("*"*80)
#         print(f"\nFunction: {func.__name__}\nConsumed memory:\nBefore: {mem_before:.2f} After: {mem_after:.2f} Difference: {(mem_after - mem_before):.2f} GB\n")
#         print("*"*80)
#         return result
#     return wrapper


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "data/cifar10", "")
DEFINE_string("output_dir", "outputs", "")

DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 1,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")
DEFINE_integer("num_epochs", 1, "How many epochs to train")
DEFINE_boolean("child_use_aux_heads", True, "")

from memory_profiler import profile

class Trainer:
    def __init__(self, dataset=None, arch=None, transfer=False):
        self.dataset = dataset
        self.arch = arch
        self.transfer = transfer
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

    @profile
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
            
            if FLAGS.child_sync_replicas:
                actual_step = global_step * FLAGS.child_num_aggregate
            else:
                actual_step = global_step

            epoch = actual_step // self.ops["num_train_batches"]
            curr_time = datetime.now()

            if global_step % FLAGS.log_every == 0:
                log_string = f"epoch={epoch:<6d}\tch_step={global_step:<6d}\tloss={child_loss:.4f}\t"
                log_string += f"lr={child_lr:.4f}\t|g|={child_grad_norm:.4f}\ttr_acc={(child_train_acc/FLAGS.batch_size):4f}\t"
                log_string += f"Time: {curr_time - start_time}"
                print(log_string)

            if actual_step % self.ops["eval_every"] == 0:
                test_images_batch, test_labels_batch = next(self.ops['child']['dataset_test'].as_numpy_iterator())
                child_test_logits = self.ops['child']['test_model'](test_images_batch)

                if FLAGS.controller_training and epoch % FLAGS.controller_train_every == 0:
                    print(f"Epoch {epoch}: Training controller")
                    images_batch, labels_batch = next(self.ops['child']['dataset_valid_shuffle'].as_numpy_iterator())
                    child_valid_rl_logits = self.ops['child']['validation_rl_model'](images_batch)

                    for ct_step in range(FLAGS.controller_train_steps * FLAGS.controller_num_aggregate):
                        controller_valid_acc = self.ops['controller']['valid_acc'](child_valid_rl_logits, labels_batch)
                        controller_loss, gn = self.controller_train_op(child_train_logits, labels)
                        controller_entropy = 0
                        lr = self.ops["controller"]["lr"]()
                        bl = self.ops["controller"]["baseline"]

                        if ct_step % FLAGS.log_every == 0:
                            curr_time = datetime.now()
                            log_string = f"ctrl_step={self.ops['controller']['train_step'].value():<6d}\tloss={controller_loss:.4f}\t"
                            log_string += f"ent={controller_entropy:.4f}\tlr={lr:.4f}\t|g|={gn:.4f}\tacc={controller_valid_acc:.4f}\t"
                            log_string += f"bl={bl.value():4f}\tTime: {curr_time - start_time}"
                            print(log_string)

                    # print("Here are 10 architectures")
                    # for _ in range(10):
                    #     arc = self.ops["controller"]["generate_sample_arc"]()
                    #     child_valid_rl_logits = self.ops['child']['validation_rl_model'](images_batch)
                    #     acc = self.ops["controller"]["valid_acc"](child_valid_rl_logits, labels_batch)
                    #     if FLAGS.search_for == "micro":
                    #         normal_arc, reduce_arc = arc
                    #         print(np.reshape(normal_arc, [-1]))
                    #         print(np.reshape(reduce_arc, [-1]))
                    #     else:
                    #         start = 0
                    #         for layer_id in range(FLAGS.child_num_layers):
                    #             if FLAGS.controller_search_whole_channels:
                    #                 end = start + 1 + layer_id
                    #             else:
                    #                 end = start + 2 * FLAGS.child_num_branches + layer_id
                    #             print(np.reshape(arc[start: end], [-1]))
                    #             start = end
                    #     print(f"val_acc={acc:<6.4f}")
                    #     print("-" * 80)
                    # print("-" * 80)

                print(f"Epoch {epoch}: Eval")
                if FLAGS.child_fixed_arc is None:
                    child_valid_acc = self.ops["eval_func"]("valid", child_valid_rl_logits, labels_batch)
                child_test_acc = self.ops["eval_func"]("test", child_test_logits, test_labels_batch)

            if epoch >= FLAGS.num_epochs:
                break

        return {
                "child_valid_acc": child_valid_acc,
                "child_test_acc": child_test_acc,
                "child_weights": self.ops["child"]["weights"],
                "controller_trainable_variables": self.ops["controller"]["trainable_variables"]
                }


# instantiation of decorator function
# @profile
# def get_ops(images, labels, arch=None, transfer=False):
#   """
#   Args:
#     images: dict with keys {"train", "valid", "test"}.
#     labels: dict with keys {"train", "valid", "test"}.
#   """
#   FLAGS = flags.FLAGS
#   assert FLAGS.search_for is not None, "Please specify --search_for"

#   # if FLAGS.search_for == "micro":
#   #   ControllerClass = MicroController
#   #   ChildClass = MicroChild
#   # else:
#   ControllerClass = MacroController
#   ChildClass = MacroChild

#   if arch is not None:
#     arch_child_weights = arch["child_weights"]
#     arch_controller_weights = arch["controller_trainable_variables"]
#   else:
#     arch_child_weights = None
#     arch_controller_weights=None
  
#   child_model = ChildClass(
#     images,
#     labels,
#     clip_mode="norm",
#     optim_algo="momentum", child_weights=arch_child_weights, transfer=transfer)

#   if FLAGS.child_fixed_arc is None:
#     controller_model = ControllerClass(
#       lstm_size=64,
#       lstm_num_layers=1,
#       lstm_keep_prob=1.0,
#       lr_dec_start=0,
#       lr_dec_every=1000000,  # never decrease learning rate
#       optim_algo="adam", controller_weights=arch_controller_weights,transfer=transfer)

#     child_train_op, child_lr, child_optimizer = child_model.connect_controller(controller_model)
#     dataset_valid_shuffle = child_model.ValidationRLShuffle(
#         child_model,
#         False)(
#             child_model.images['valid_original'],
#             child_model.labels['valid_original'])
#     controller_train_op, controller_lr, controller_optimizer = controller_model.build_trainer(
#         child_model,
#         child_model.ValidationRL())

#     controller_ops = {
#       "train_step": controller_model.train_step, # tf.Variable
#       # MacroController.generate_sample_arc() -> sample_arc
#       # MicroController.generate_sample_arc() -> normal_arc, reduce_arc
#       "generate_sample_arc": controller_model.generate_sample_arc,
#       'loss': controller_model.loss, # Controller.loss(child_logits, y_valid_shuffle) -> loss
#       "train_op": controller_train_op, # Controller.train_op(loss, vars) -> iteration_num
#       "lr": controller_lr, # learning_rate() -> learning_rate
#       'trainable_variables': controller_model.trainable_variables(),
#       "valid_acc": controller_model.valid_acc, # Controller.valid_acc(child_logits, y_valid_shuffle) -> valid_acc
#       "optimizer": controller_optimizer, # framework.Optimizer
#       "baseline": controller_model.baseline, # tf.Variable
#       "entropy": lambda: controller_model.current_entropy,
#     }
#     child_valid_rl_model = child_model.valid_rl_model
#   else:
#     assert not FLAGS.controller_training, (
#       "--child_fixed_arc is given, cannot train controller")
#     child_train_op, child_lr, child_optimizer = child_model.connect_controller(None)
#     dataset_valid_shuffle = None
#     controller_ops = None
#     child_valid_rl_model = None
  
#   return {
#     "child": {
#       'generate_train_losses': child_model.generate_train_losses, # Child.generate_train_losses(images, labels) -> logits, loss, train_loss, train_acc
#       'test_model': child_model.test_model, # Child.test_model(images) -> child_logits
#       'validation_model': child_model.valid_model, # Child.valid_model(images) -> child_logits
#       'validation_rl_model': child_valid_rl_model, # Child.valid_rl_model(images) -> child_valid_rl_logits
#       'global_step': child_model.global_step, # tf.Variable
#       'dataset': child_model.dataset,
#       'dataset_test': child_model.dataset_test,
#       'dataset_valid_shuffle': dataset_valid_shuffle, # tf.Dataset
#       "loss": child_model.loss, # Child.loss(child_logits) -> loss
#       # MacroChild.loss(child_logits) -> train_loss
#       # MicroChild.loss(child_logits, child_aux_logits) -> train_loss
#       "train_loss": child_model.train_loss,
#       "train_op": child_train_op, # Child.train_op(train_loss, vars) -> iteration_num
#       "lr": child_lr, # Child.learning_rate() -> learning_rate
#       'trainable_variables': child_model.trainable_variables(),
#       "optimizer": child_optimizer, # framework.Optimizer
#       "num_train_batches": child_model.num_train_batches,
#       "weights": child_model.weights.weight_map, #PRUEBAA!
#     },
#     "controller": controller_ops,
#     "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
#     "eval_func": child_model.eval_once,
#     "num_train_batches": child_model.num_train_batches,
#   }


# instantiation of decorator function
# @profile
# def train(arch=None,transfer=False):
#   FLAGS = flags.FLAGS
#   # if FLAGS.child_fixed_arc is None:
#   #   images, labels = read_data(FLAGS.data_path)
#   # else:
#   #   images, labels = read_data(FLAGS.data_path, num_valids=0)
  
#   with open('/workspace/Proof-of-concept/data/cifar10/all_data.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     data = pickle.load(f)
  
#   images, labels = data["images"], data["labels"]
  
#   ops = get_ops(images, labels, arch=arch, transfer=transfer)
#   del images, labels, data
#   gc.collect()
#   batch_iterator = None

#   print("-" * 80)
#   print("Starting session")
#   start_time = datetime.now() #time.time()

#   @fw.function
#   def child_train_op(images, labels):
#       with fw.GradientTape() as tape:
#           child_train_logits, child_loss, child_train_loss, child_train_acc = ops['child']['generate_train_losses'](images, labels)
#           child_grad_norm, child_grad_norm_list, _ = ops['child']['train_op'](child_loss, ops['child']['trainable_variables'], tape)
#       return child_train_logits, child_loss, child_train_acc, child_grad_norm

#   @fw.function
#   def controller_train_op(child_train_logits, labels):
#       with fw.GradientTape() as tape:
#           ops["controller"]["generate_sample_arc"]()
#           controller_loss = ops['controller']['loss'](child_train_logits, labels)
#           gn, gn_list, _ = ops['controller']['train_op'](controller_loss, ops['controller']['trainable_variables'], tape)
#           return controller_loss, gn

#   architecture = {}
#   while True:
#       if batch_iterator is None:
#           batch_iterator = ops['child']['dataset'].as_numpy_iterator()
#       try:
#           images, labels = batch_iterator.__next__()
#       except StopIteration:
#           batch_iterator = ops['child']['dataset'].as_numpy_iterator()
#           images, labels = batch_iterator.__next__()
#       child_train_logits, child_loss, child_train_acc, child_grad_norm = child_train_op(images, labels)
#       child_lr = ops['child']['lr']()
#       global_step = ops["child"]["global_step"].value()
#       if FLAGS.child_sync_replicas:
#           actual_step = global_step * FLAGS.child_num_aggregate
#       else:
#           actual_step = global_step
#       epoch = actual_step // ops["num_train_batches"]
#       curr_time = datetime.now() #time.time()
#       if global_step % FLAGS.log_every == 0:
#           log_string = ""
#           log_string += "epoch={:<6d}".format(epoch)
#           log_string += "ch_step={:<6d}".format(global_step)
#           log_string += " loss={:<8.6f}".format(child_loss)
#           log_string += " lr={:<8.4f}".format(child_lr)
#           log_string += " |g|={:<8.4f}".format(child_grad_norm)
#           log_string += " tr_acc={:<3d}/{:>3d}".format(
#               child_train_acc, FLAGS.batch_size)
#           log_string += " Time: {}".format(curr_time - start_time)
#           print(log_string)
          
#       if actual_step % ops["eval_every"] == 0:
#         test_images_batch, test_labels_batch = ops['child']['dataset_test'].as_numpy_iterator().__next__()
#         child_test_logits = ops['child']['test_model'](test_images_batch)
#         if (FLAGS.controller_training and
#             epoch % FLAGS.controller_train_every == 0):
#           print(("Epoch {}: Training controller".format(epoch)))
#           images_batch, labels_batch = ops['child']['dataset_valid_shuffle'].as_numpy_iterator().__next__()
#           child_valid_rl_logits = ops['child']['validation_rl_model'](images_batch)
#           for ct_step in range(FLAGS.controller_train_steps *
#                                 FLAGS.controller_num_aggregate):
#             controller_valid_acc = ops['controller']['valid_acc'](child_valid_rl_logits, labels_batch)
#             controller_loss, gn = controller_train_op(child_train_logits, labels)
#             controller_entropy = 0
#             lr = ops["controller"]["lr"]()
#             bl = ops["controller"]["baseline"]

#             if ct_step % FLAGS.log_every == 0:
#               curr_time = datetime.now() #time.time()
#               log_string = ""
#               log_string += "ctrl_step={:<6d}".format(ops["controller"]["train_step"].value())
#               log_string += " loss={:<7.3f}".format(controller_loss)
#               log_string += " ent={:<5.2f}".format(controller_entropy)
#               log_string += " lr={:<6.4f}".format(lr)
#               log_string += " |g|={:<8.4f}".format(gn)
#               log_string += " acc={:<6.4f}".format(controller_valid_acc)
#               log_string += f' bl={bl.value()}'
#               log_string += " Time: {}".format(curr_time - start_time)
#               print(log_string)

#           print("Here are 10 architectures")

#           for _ in range(10):
#             arc = ops["controller"]["generate_sample_arc"]()
#             child_valid_rl_logits = ops['child']['validation_rl_model'](images_batch)
#             acc = ops["controller"]["valid_acc"](child_valid_rl_logits, labels_batch)
#             if FLAGS.search_for == "micro":
#               normal_arc, reduce_arc = arc
#               print((np.reshape(normal_arc, [-1])))
#               print((np.reshape(reduce_arc, [-1])))
#             else:
#               start = 0
#               for layer_id in range(FLAGS.child_num_layers):
#                 if FLAGS.controller_search_whole_channels:
#                   end = start + 1 + layer_id
#                 else:
#                   end = start + 2 * FLAGS.child_num_branches + layer_id
#                 print((np.reshape(arc[start: end], [-1])))
#                 start = end
#             print(("val_acc={:<6.4f}".format(acc)))
#             print(("-" * 80))
            
#           print(("-" * 80))

#         print(("Epoch {}: Eval".format(epoch)))
#         if FLAGS.child_fixed_arc is None:
#           child_valid_acc = ops["eval_func"]("valid", child_valid_rl_logits, labels_batch)
#         child_test_acc = ops["eval_func"]("test", child_test_logits, test_labels_batch)

#         del images, labels, child_train_logits, child_loss, child_train_acc, child_grad_norm, \
#           test_images_batch, test_labels_batch, images_batch, labels_batch, 
#         gc.collect()
        
#       if epoch >= FLAGS.num_epochs:
#         architecture = {"child_valid_acc": child_valid_acc,
#                         "child_test_acc": child_test_acc,
#                         "child_weights": ops["child"]["weights"],
#                         "controller_trainable_variables": ops["controller"]["trainable_variables"]
#                         }
#         del ops
#         gc.collect()
#         K.clear_session()
#         break
#   return architecture


# def main(arch=None,transfer=False):
#   FLAGS = flags.FLAGS
#   print(("-" * 80))
#   if not os.path.isdir(FLAGS.output_dir):
#     print(("Path {} does not exist. Creating.".format(FLAGS.output_dir)))
#     os.makedirs(FLAGS.output_dir)
#   elif FLAGS.reset_output_dir:
#     print(("Path {} exists. Remove and remake.".format(FLAGS.output_dir)))
#     shutil.rmtree(FLAGS.output_dir)
#     os.makedirs(FLAGS.output_dir)

#   print(("-" * 80))
#   log_file = os.path.join(FLAGS.output_dir, "stdout")
#   print(("Logging to {}".format(log_file)))
#   sys.stdout = Logger(log_file)

#   print_user_flags()
#   architecture = train(arch=arch,transfer=transfer)
#   return architecture
  


# if __name__ == "__main__":
#     #app.run(main)
#     main(arch=None,transfer=False)

