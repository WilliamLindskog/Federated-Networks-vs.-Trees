import torch
import pandas as pd

from torch.utils.data import Dataset

from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
from omegaconf import DictConfig

class TabularDataset(Dataset):
    """Tabular dataset."""
    
    def __init__(self, data: pd.DataFrame, target: str, device: str) -> None:
        """Initialize TabularDataset.
        
        Parameters
        ----------
        data : DataFrame
            Data.
        target : str
            Target.
        """
        self.data = data.astype('float32')
        self.target = target.astype('float32')
        self.device = device

    def __len__(self) -> int:
        """Get length of dataset.
        
        Returns
        -------
        int
            Length of dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get item from dataset.
        
        Parameters
        ----------
        idx : int
            Index.
        
        Returns
        -------
        Tuple[Any, Any]
            Tuple of data and target.
        """
        data = self.data.iloc[idx]
        target = self.target.iloc[idx]

        # Set to tensor
        data = torch.tensor(data.values)
        # target is one value
        target = torch.tensor(target)
        return data.to(self.device), target.to(self.device)

def dataset_prepare(name: str, dataset: pd.DataFrame, cfg: DictConfig) -> Any:
    """Prepare dataset.
    
    Parameters
    ----------
    name : str
        Name of dataset.
    dataset : Union[DataFrame, Any]
        Dataset.
    
    Returns
    -------
    Tuple[str, Any]
        Tuple of dataset name and dataset.
    """

    # Get dataset path
    if name in ['smoking', 'heart', 'lumpy', 'machine', 'insurance']:
        dataset = standard_preprocessing(dataset)
    elif name == 'femnist':
        dataset = femnist_preprocessing(dataset, cfg)
    elif name == 'synthetic':
        pass
    else:
        raise NotImplementedError(f'Unknown dataset {name}')
    
    return dataset

def standard_preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
    """Processing insurance dataset."""
    # Encode categorical features
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print("Encoding categorical feature: ", col)
            dataset[col] = dataset[col].astype('category').cat.codes
            # Set to int64
            dataset[col] = dataset[col].astype('int64')
            
    return dataset

def femnist_preprocessing(dataset: pd.DataFrame, cfg: DictConfig) -> None:
    print("FEMNIST preprocessing")
    # remove all columns where all values are the same
    print(dataset.shape)
    dataset = dataset.loc[:, (dataset != dataset.iloc[0]).any()]
    print(dataset.shape)
    # remove all columns where 99% of values are the same
    dataset = dataset.loc[:, (dataset == dataset.iloc[0]).sum() < 0.99*len(dataset)]
    print(dataset.shape)

    if cfg.dataset.iid:
        dataset.drop(columns=['user'], inplace=True)

    # Encode categorical features
    return standard_preprocessing(dataset)

def smoking_processing(dataset: pd.DataFrame, cfg: DictConfig) -> None:
    """Processing smoking dataset."""
    
    # Drop ID and oral column
    dataset.drop('ID', axis=1, inplace=True)
    dataset.drop('oral', axis=1, inplace=True)

    return standard_preprocessing(dataset)