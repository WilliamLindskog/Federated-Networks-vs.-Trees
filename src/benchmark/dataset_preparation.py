from typing import List, Tuple, Dict, Union, Any
from pandas import DataFrame

from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    """Tabular dataset."""
    
    def __init__(self, data: DataFrame, target: str) -> None:
        """Initialize TabularDataset.
        
        Parameters
        ----------
        data : DataFrame
            Data.
        target : str
            Target.
        """
        self.data = data
        self.target = target

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
        return data, target

def dataset_prepare(name: str, dataset: Union[DataFrame, Any]) -> Any:
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
        smoking_processing(dataset)
    else:
        raise NotImplementedError(f'Unknown dataset {name}')
    
    return dataset

def smoking_processing(dataset: DataFrame) -> None:
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