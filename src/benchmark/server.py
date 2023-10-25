from flwr.common import NDArrays, Scalar

from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from collections import OrderedDict

from pathlib import Path
from src.benchmark.utils import get_dataset, get_server_dataset

from omegaconf import DictConfig

FL_BENCH_ROOT = Path(__file__).resolve().parent.parent.parent

def get_evaluate_fn(cfg: DictConfig, model):
    """Return an evaluation function for server-side evaluation."""

    # read server data from tmp folder
    test_loader = get_server_dataset(cfg)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # add parameters to model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = model.evaluate(test_loader)
        return loss, {"accuracy": accuracy}

    return evaluate