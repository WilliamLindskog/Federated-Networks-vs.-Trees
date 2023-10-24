from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.benchmark.dataset_preparation import TabularDataset
from src.benchmark.constants import SMOKING_COLUMNS_TO_SCALE

from torch.utils.data import DataLoader, Dataset

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
    
def _task_assertions(cfg: DictConfig) -> None:
    """Task assertions.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    """
    if cfg.dataset.name == 'smoking':
        assert cfg.task == 'binary', \
            "Task must be classification for smoking dataset."
        assert cfg.dataset.num_classes == 2, \
            "Number of classes must be 2 for smoking dataset."
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

    if name == 'smoking':
        cfg.dataset.num_classes = 2
        cfg.task = 'binary'
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
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

def get_centralized_dataset(cfg: DictConfig) -> Any:
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
    csv = True if name in ['smoking'] else False

    columns_to_scale = SMOKING_COLUMNS_TO_SCALE

    tmp_path = "./tmp"
    if csv:
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