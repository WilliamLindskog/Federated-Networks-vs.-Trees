from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.benchmark.dataset_preparation import TabularDataset
from src.benchmark.constants import SMOKING_COLUMNS_TO_SCALE

from torch.utils.data import DataLoader, Dataset

import pickle
from pathlib import Path
from flwr.server.history import History
from secrets import token_hex
import matplotlib.pyplot as plt
import numpy as np

def get_on_fit_config(cfg: DictConfig):
    """Get on fit config."""
    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
        _ = server_round
        return fit_config

    return fit_config_fn

def initial_assertions(cfg: DictConfig) -> None:
    """Initial assertions.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    """
    assert len(cfg.model.hidden_dim) == cfg.model.num_layers, \
        "Number of hidden sizes must be equal to number of layers."
    _task_assertions(cfg)

    assert cfg.clients_per_round <= cfg.num_clients, \
        "Number of clients per round must be less than or equal to number of clients."
    
def _task_assertions(cfg: DictConfig) -> None:
    """Task assertions.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    """
    if cfg.dataset.name in ['smoking', 'heart', 'lumpy', 'machine']:
        assert cfg.task == 'binary', \
            "Task must be classification for smoking dataset."
        assert cfg.dataset.num_classes == 2, \
            "Number of classes must be 2 for smoking dataset."
    elif cfg.dataset.name == 'femnist':
        assert cfg.task == 'multiclass', \
            "Task must be multiclass for femnist dataset."
        assert cfg.dataset.num_classes == 62, \
            "Number of classes must be 62 for femnist dataset."
    elif cfg.dataset.name == 'synthetic':
        pass
    elif cfg.dataset.name == 'insurance':
        assert cfg.task == 'regression', \
            "Task must be regression for insurance dataset."
        assert cfg.dataset.num_classes == 1, \
            "Number of classes must be 1 for insurance dataset."
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} not implemented.")

def set_model_target(name: str, cfg: DictConfig) -> DictConfig:
    """ Set model target.
    
    Parameters
    ----------
    name : str
        Name of model.
    cfg : DictConfig
        Configuration file.

    Returns
    -------
    DictConfig
        Configuration file.
    """ 
    if name == 'mlp':
        cfg.model._target_ = "src.benchmark.models.mlp.MLP"
    else:
        raise NotImplementedError(f"Model {name} not implemented.")
    return cfg

def set_dataset_task(name: str, cfg: DictConfig) -> DictConfig:
    """Set dataset task.
    
    Parameters
    ----------
    name : str
        Name of dataset.
    cfg : DictConfig
        Configuration file.

    Returns
    -------
    DictConfig
        Configuration file.
    """

    if name in ['smoking', 'heart', 'lumpy', 'machine']:
        cfg.dataset.num_classes = 2
        cfg.task = 'binary'
    elif name == 'femnist':
        cfg.dataset.num_classes = 62
        cfg.task = 'multiclass'
    elif name == 'synthetic':
        pass
    elif name == 'insurance':
        cfg.dataset.num_classes = 1
        cfg.task = 'regression'
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")
    return cfg

def get_model(cfg: DictConfig) -> Any:
    """Get model.
    
    Parameters
    ----------
    name : str
        Name of model.
    **kwargs
        Model parameters.
    
    Returns
    -------
    Any
        Model.
    """
    model = instantiate(OmegaConf.to_container(cfg, resolve=True))
    return model

def get_target_name(dataset_name: str) -> str:
    """Get target name.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset.
    """

    if dataset_name == 'smoking':
        return 'smoking'
    elif dataset_name == 'heart':
        return 'target'
    elif dataset_name == 'lumpy':
        return 'lumpy'
    elif dataset_name == 'machine':
        return 'Target'
    elif dataset_name in ['femnist','synthetic']:
        return 'y'
    elif dataset_name == 'insurance':
        return 'charges'
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

def get_dataset(cfg: DictConfig, cid: str = None, server = False) -> Any:
    """Get centralized dataset.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    
    Returns
    -------
    Any
        Dataset.
    """
    name, device = cfg.name, cfg.device
    csv = True if name in [
        'smoking', 'heart', 'lumpy', 'machine', 'femnist', 'synthetic', 'insurance'
    ] else False

    if name not in ['femnist', 'synthetic']:
        columns_to_scale = SMOKING_COLUMNS_TO_SCALE[name]
    else:
        columns_to_scale = None

    tmp_path = "./tmp"
    if csv:
        if cid is not None:
            data = pd.read_csv(f'{tmp_path}/data_{cid}.csv')
        else:
            data = pd.read_csv(f'{tmp_path}/data.csv')
        X = data.drop(columns=[get_target_name(name)])
        y = data[get_target_name(name)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=cfg.test_frac, 
            random_state=42,
        )
        scaler = StandardScaler()
        if columns_to_scale is not None:
            X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
            X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
        train_dataset = TabularDataset(X_train, y_train, device=device)
        test_dataset = TabularDataset(X_test, y_test, device=device)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False,
        )

    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")
    
    return train_loader, test_loader


def get_server_dataset(cfg: DictConfig) -> Any:
    """Get centralized dataset."""
    name, device = cfg.name, cfg.device
    csv = True if name in [
        'smoking', 'heart', 'lumpy', 'machine', 'femnist', 'synthetic', 'insurance'
    ] else False

    if name not in ['femnist', 'synthetic']:
        columns_to_scale = SMOKING_COLUMNS_TO_SCALE[name]
    else:
        columns_to_scale = None

    tmp_path = "./tmp"
    if csv:
        data = pd.read_csv(f'{tmp_path}/server_data.csv')
        X = data.drop(columns=[get_target_name(name)])
        y = data[get_target_name(name)]

        scaler = StandardScaler()
        if columns_to_scale is not None:
            X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale]) 
        dataset = TabularDataset(X, y, device=device)
        loader = DataLoader(
            dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False,
        )
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")
    return loader

def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
    metric_type: Optional[str] = "centralized",
    regression: Optional[bool] = False,
) -> None:
    """Plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )

    if regression:
        _, values = zip(*metric_dict["loss"])
        values = tuple(x.item() for x in values)
    else:
        _, values = zip(*metric_dict["accuracy"])

    if metric_type == "centralized":
        rounds_loss, values_loss = zip(*hist.losses_centralized)
        # make tuple of normal floats instead of tensors
        values_loss = tuple(x.item() for x in values_loss)
    else:
        # let's extract decentralized loss (main metric reported in FedProx paper)
        rounds_loss, values_loss = zip(*hist.losses_distributed)


    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    if metric_type == "centralized":
        axs[1].plot(np.asarray(rounds_loss), np.asarray(values))    
    else:
        axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")

    if regression:
        axs[1].set_ylabel("Loss")
    else:
        axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    default_filename: Optional[str] = "results.pkl",
) -> None:
    """Save results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a random suffix to the file name."""
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        if default_filename is None:
            return path_
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")
    # data = {"history": history, **extra_results}
    data = {"history": history}
    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def empty_dir(path: Path) -> None:
    """Empty directory.
    
    Parameters
    ----------
    path : Path
        Path to directory.
    """
    for file in path.iterdir():
        if file.is_file():
            file.unlink()
        else:
            empty_dir(file)
            file.rmdir()