import json
from datetime import datetime

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from ultralytics import YOLO
from torch.utils.tensorboard.writer import SummaryWriter
import torch

class CustomFedAvg(FedAvg):
    """
    Custom strategy from Federated Average strategy
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Name the data in tensorboard
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_dir = "runs/flower_" + name
        self.writer = SummaryWriter(log_dir=log_dir)

        self.results_to_save = {}

        self.class_names = ['bishop', 'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']

    def aggregate_fit(self,
                      server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """
        Save intermediate wrights of the global model
        """
        global PROJECT_PATH
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # Load actual model state
        ndarrays_parameters = parameters_to_ndarrays(parameters_aggregated)
        generic_model = YOLO(f"{PROJECT_PATH}/yolo11s_generic_model.pt")
        model = set_model(generic_model, ndarrays_parameters)
        torch.save(model.model.state_dict(), f"{PROJECT_PATH}/global_model_weights/global_model_round_{server_round}.pt")
        
        return parameters_aggregated, metrics_aggregated
    