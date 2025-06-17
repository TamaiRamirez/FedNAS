import socket
import pickle
from main import Trainer
import os
import sys
import copy
import shutil
import numpy as np
from src.utils import Logger
import random
import gc
import pandas as pd
import time
gc.enable()

# Define the client class
class Client:
    data = b""
    packet = None
    result = None
    i=1
    def __init__(self, host='localhost', port=12345, clients = 2):
        self.host = host
        self.port = port
        self.close_socket = False
        self.values = {}
        self.data = b""
        self.packet = None
        self.result = None
        self.round=1
        self.trainer = None
        self.dataset = None
        self.num_clients = clients
        with open('/workspace/Proof-of-concept/data/cifar10/all_data.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
        # self.dataset = self.random_dataset(self.dataset)
        
        
    def random_dataset(self, dataset = {}):
        for key in dataset["images"]:
            # Get the length of the current list
            length = len(dataset["images"][key])
            # Determine the number of elements to sample
            sample_size = length // self.num_clients
            # Get the indices to sample
            random.seed(42)
            indices = random.sample(range(length), sample_size)
            
            # Select the elements corresponding to the sampled indices
            dataset["images"][key] = [dataset["images"][key][i] for i in indices]
            dataset["labels"][key] = [dataset["labels"][key][i] for i in indices]
    
        return dataset
        
    def start(self):
        # Start the client socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        print("Connected to the server.")
        try:
            while True:
                self.data=b""
                while True:
                    self.packet = client_socket.recv(4096)
                    if str.encode("client") in self.packet:
                        index = self.packet.find(str.encode("client"))
                        # client_name = packet[index:index+8]
                        self.packet = self.packet[index+8:]
                    
                    if str.encode("_best_model") in self.packet:
                        index = self.packet.find(str.encode("_best_model"))
                        self.packet = self.packet[index+11:]
                        # output_dir = "/workspace/Proof-of-concept/outputs_best_model/"
                        # log_file = os.path.join(output_dir, client_name.decode("utf-8"))
                        # print(("Logging to {}".format(log_file)))
                        # sys.stdout = Logger(log_file)
                    
                    if str.encode("_weights_avg") in self.packet:
                        index = self.packet.find(str.encode("_weights_avg"))
                        self.packet = self.packet[index+12:]
                        # output_dir = "/workspace/Proof-of-concept/outputs_weights_avg/"
                        # log_file = os.path.join(output_dir, client_name.decode("utf-8"))
                        # print(("Logging to {}".format(log_file)))
                        # sys.stdout = Logger(log_file)

                    self.data+=self.packet
                    if str.encode("finalizar") in self.packet:
                        self.packet.replace(str.encode("finalizar"),b'')
                        break
                    
                    if str.encode("close") in self.packet:
                        self.packet.replace(str.encode("close"),b'')
                        self.close_socket = True
                        print("Connection closed.")
                        break
                if self.data != b"" and not self.close_socket:
                    self.values = pickle.loads(self.data)  
                    # self.dataset = self.random_dataset(self.dataset)
                    self.trainer = Trainer(copy.deepcopy(self.dataset),self.values["architecture"],self.values["transfer"])
                    gc.collect()
                    self.result = self.trainer.train()
                    self.result = pickle.dumps(self.result)
                    client_socket.sendall(self.result)
                    del self.data, self.values, self.packet, self.result, self.trainer
                    gc.collect()
                    client_socket.sendall(str.encode("finalizar"))
                    print(f"\nRESULT SENT\n")
                    print(f"Round {self.round} Finished")
                    self.round+=1
                    print("-"*80)
                    print("-"*80)
                    # for name in dir():
                    #     if not name.startswith('_'):
                    #         del globals()[name]

                    # for name in dir():
                    #     if not name.startswith('_'):
                    #         del locals()[name]
                else:
                    client_socket.close()

        finally:
            client_socket.close()
            print("Connection closed.")

# Start the client
if __name__ == "__main__":
    client = Client(clients=2)
    client.start()
