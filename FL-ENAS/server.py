import socket
import threading
import pickle
from datetime import datetime, timedelta
import src.framework as fw
import os
import sys
import copy
import shutil
import numpy as np
from src.utils import Logger
import psutil
import gc
gc.enable()
# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):

        mem_before = process_memory() / (1024*1024*1024)
        result = func(*args, **kwargs)
        mem_after = process_memory() / (1024*1024*1024)
        print("*"*80)
        print(f"\nFunction: {func.__name__}\nConsumed memory:\nBefore: {mem_before:.2f} After: {mem_after:.2f} Difference: {(mem_after - mem_before):.2f} GB\n")
        print("*"*80)
        return result
    return wrapper

# Define the server class
class Server:
    def __init__(self, host='localhost', port=12345, num_clients=2, rounds=5, method="best_model"):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.rounds = rounds
        self.method = method
        self.client_sockets = []
        self.lock = threading.Lock()
        self.W = {"child_weights": None, "controller_trainable_variables": None}
        self.transfer = False
        self.data = None
        self.results = None
        self.valid_acc = []
        self.test_acc = []
    
    def start(self):
        # Start the server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(self.num_clients)
        # output_dir = f"/workspace/Proof-of-concept/outputs_{self.method}/"
        # shutil.rmtree(output_dir)
        # os.makedirs(output_dir)
        # log_file = os.path.join(output_dir, "server")
        # print(("Logging to {}".format(log_file)))
        # sys.stdout = Logger(log_file)
        
        print(f"Server started, waiting for {self.num_clients} clients to connect...")
        for _ in range(self.num_clients):
            client_socket, addr = server_socket.accept()
            self.client_sockets.append(client_socket)
            print(f"Client {addr} connected.")

        print(f"\nINITIAL TIME: {datetime.now()+timedelta(hours=2)}")
        print(f"\nMODE: {self.method}")
        start_time = datetime.now()
        for round_num in range(self.rounds):
            print(f"\nStarting round {round_num + 1} of {self.rounds}...")
                
            self.data = pickle.dumps({"architecture": self.W, "transfer":self.transfer})
            self.broadcast_value(self.data,round_num)
            self.results = self.collect_results()
            self.W = self.weights_aggregate(self.results,method=self.method)
            self.valid_acc.append(self.W["child_valid_acc"])
            self.test_acc.append(self.W["child_test_acc"])
            print(f"\nRound {round_num + 1} completed.")
            curr_time = datetime.now() #time.time()
            print("Time: {}\n".format(curr_time - start_time))
            self.transfer = True
            if round_num == self.rounds-1:
                acc = self.W["child_test_acc"]
                print(f"Accuracy Test final: {acc}")
                print(f"FINISH TIME: {datetime.now()+timedelta(hours=2)}")
                
                # with open(output_dir+'acuraccies.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                #     pickle.dump({"valid_acc":valid_acc,"test_acc":test_acc}, f)
                
                with open('/workspace/Proof-of-concept/acuraccies.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump({"valid_acc":self.valid_acc,"test_acc":self.test_acc}, f)
                print("-"*80)
                print("-"*80)
                print("-"*80)
                print("-"*80)
                print()
 
        self.close_connections()

    def weights_aggregate(self, results, method="best_model"):
        if method == "best_model":
            best_model={}
            best_acc=-1.0
            for result in results:
                if result["child_valid_acc"] > best_acc:
                    best_model=result
            return best_model
        elif method == "weights_avg":
            max_acc = -1.0
            best_controller = {}
            for result in results:
                if result["child_valid_acc"] > max_acc:
                    max_acc=result["child_valid_acc"]
                    best_controller = result["controller_trainable_variables"]
                    
            for layer in results[0]["child_weights"]:
                avg_w=fw.zeros(result["child_weights"][layer].get_shape())
                for result in results:
                    avg_w+=(result["child_valid_acc"]/max_acc)*result["child_weights"][layer].value()
                results[0]["child_weights"][layer] = fw.Variable(avg_w/self.num_clients,name=layer+":0",trainable=True)
            results[0]["controller_trainable_variables"] = best_controller
            del best_controller, max_acc, avg_w
            gc.collect()
            return results[0]
            
    def broadcast_value(self, value, round_num):
        # Broadcast a value to all clients
        i=1
        for client_socket in self.client_sockets:
            if round_num == 0:
                client_socket.sendall(str.encode(f"client_{i}_{self.method}"))
                i+=1
            client_socket.sendall(copy.deepcopy(value))
            client_socket.sendall(str.encode("finalizar"))

    def collect_results(self):
        results = []
        for client_socket in self.client_sockets:
            data = b""
            while True:
                packet = client_socket.recv(4096)
                data += packet
                if str.encode("finalizar") in packet:
                    packet.replace(str.encode("finalizar"),b'')
                    break
            result = pickle.loads(data)
            data = None
            results.append(result)
        return results

    def close_connections(self):
        for client_socket in self.client_sockets:
            client_socket.sendall(str.encode("close"))
            client_socket.close()
        print("All connections closed.")

# Start the server
if __name__ == "__main__":
    server = Server(num_clients=2, rounds=31, method="weights_avg")
    server.start()
