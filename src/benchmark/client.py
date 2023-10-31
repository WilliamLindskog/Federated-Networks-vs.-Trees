"""Client implementation - can call FedPer and FedAvg clients."""
import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pandas as pd

from src.benchmark.utils import get_dataset

PROJECT_DIR = Path(__file__).parent.parent.absolute()

class BaseClient(NumPyClient):
    def __init__(self, cid, model, trainloader, valloader, config):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config

        self.task = config.task

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metric_dict = self.model.train(self.trainloader)
        if 'accuracy' in metric_dict.keys(): 
            return self.get_parameters(config), len(self.trainloader), {'accuracy': metric_dict['accuracy']}
        else:
            return self.get_parameters(config), len(self.trainloader), {'loss': metric_dict['loss']}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metric = self.model.evaluate(self.valloader)
        if self.task != 'regression':
            print(f"[Client {self.cid}] evaluate, accuracy: {metric}")
            return float(loss), len(self.valloader), {"accuracy": float(metric)}
        print(f"[Client {self.cid}] evaluate, loss: {loss}")
        return float(loss), len(self.valloader), {"loss": float(loss)}


def get_client_fn_simulation(
    config: DictConfig,
    model: Any,
) -> Callable[[str], BaseClient]:
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    model : DictConfig
        The overall configuration.
    cleint_state_save_path : str
        The path to save the client state.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    assert config.model.name.lower() in [
        "mlp",
    ], f"Model {config.model.name} not implemented"

    def client_fn(cid: str) -> BaseClient:
        """Create a Flower client representing a single organization."""

        # 1. Load data
        train_loader, val_loader = get_dataset(config.dataset, cid=cid)

        return BaseClient(cid, model, train_loader, val_loader, config)

    return client_fn
