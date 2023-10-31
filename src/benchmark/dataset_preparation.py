from typing import List, Tuple, Dict, Union, Any
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch
from geopy.geocoders import Nominatim

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

def dataset_prepare(name: str, dataset: Union[pd.DataFrame, Any]) -> Any:
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
    if name == 'smoking':
        dataset = smoking_processing(dataset)
    elif name == 'heart':
        dataset = heart_preprocessing(dataset)
    elif name == 'lumpy':
        dataset = lumpy_preprocessing(dataset)
    elif name == 'machine':
        dataset = machine_preprocessing(dataset)
    elif name == 'femnist':
        dataset = femnist_preprocessing(dataset)
    elif name == 'synthetic':
        pass
    elif name == 'insurance':
        dataset = insurance_preprocessing(dataset)
    else:
        raise NotImplementedError(f'Unknown dataset {name}')
    
    return dataset

def insurance_preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
    """Processing insurance dataset."""
    # Encode categorical features
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print("Encoding categorical feature: ", col)
            dataset[col] = dataset[col].astype('category').cat.codes
            # Set to int64
            dataset[col] = dataset[col].astype('int64')
            
    return dataset

def femnist_preprocessing(dataset: pd.DataFrame) -> None:
    print("FEMNIST preprocessing")
    # remove all columns where all values are the same
    print(dataset.shape)
    dataset = dataset.loc[:, (dataset != dataset.iloc[0]).any()]
    print(dataset.shape)
    # remove all columns where 99% of values are the same
    dataset = dataset.loc[:, (dataset == dataset.iloc[0]).sum() < 0.99*len(dataset)]
    print(dataset.shape)

    # Encode categorical features
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print("Encoding categorical feature: ", col)
            dataset[col] = dataset[col].astype('category').cat.codes
            # Set to int64
            dataset[col] = dataset[col].astype('int64')

    return dataset

def machine_preprocessing(dataset: pd.DataFrame) -> None:
    """Processing machine dataset."""

    # Encode categorical features
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print("Encoding categorical feature: ", col)
            dataset[col] = dataset[col].astype('category').cat.codes
            # Set to int64
            dataset[col] = dataset[col].astype('int64')
    
    return dataset

def lumpy_preprocessing(dataset: pd.DataFrame) -> None:
    """ Lumpy skin data preprocessing. """

    # Encode categorical features
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print("Encoding categorical feature: ", col)
            dataset[col] = dataset[col].astype('category').cat.codes
            # Set to int64
            dataset[col] = dataset[col].astype('int64')

    return dataset

def heart_preprocessing(dataset: pd.DataFrame) -> None:
    """Processing heart disease dataset."""

    # Encode categorical features
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print("Encoding categorical feature: ", col)
            dataset[col] = dataset[col].astype('category').cat.codes
            # Set to int64
            dataset[col] = dataset[col].astype('int64')

    return dataset

def smoking_processing(dataset: pd.DataFrame) -> None:
    """Processing smoking dataset."""
    
    # Drop ID and oral column
    dataset.drop('ID', axis=1, inplace=True)
    dataset.drop('oral', axis=1, inplace=True)

    # Encode categorical features
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print("Encoding categorical feature: ", col)
            dataset[col] = dataset[col].astype('category').cat.codes
            # Set to int64
            dataset[col] = dataset[col].astype('int64')
    
    return dataset