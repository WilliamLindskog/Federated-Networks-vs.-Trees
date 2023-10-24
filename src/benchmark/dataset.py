import sys
from omegaconf import DictConfig
from pathlib import Path

import pandas as pd

from typing import Tuple, Any, Union, Optional
from src.benchmark.dataset_preparation import dataset_prepare

WORKING_DIR = Path(__file__).resolve().parent
FL_BENCH_ROOT = WORKING_DIR.parent.parent

sys.path.append(FL_BENCH_ROOT.as_posix())

def get_dataset_path(name: str) -> Tuple[Path, bool]:
    """Get dataset path.
    
    Parameters
    ----------
    name : str
        Name of dataset.
    
    Returns
    -------
    Path
        Path to dataset.
    bool
        Is dataset .csv?
    """
    if name == 'smoking':
        return FL_BENCH_ROOT / 'data' / 'smoking' / 'smoking.csv', True
    else:
        raise NotImplementedError(f'Unknown dataset {name}')

def get_dataset(name: str) -> Tuple[Any, bool]:
    """Get dataset.
    
    Parameters
    ----------
    name : str
        Name of dataset.
    
    Returns
    -------
    Tuple[Any, bool]
        Tuple of dataset and is dataset .csv?
    """
    csv = False
    if name == 'smoking':
        data = pd.read_csv(FL_BENCH_ROOT / 'data' / 'smoking' / 'smoking.csv')
        csv = True
    else:
        raise NotImplementedError(f'Unknown dataset {name}')

    return data, csv

def dataset_main(cfg: DictConfig) -> Optional[DictConfig]:
    """Main function for dataset."""
    # Remove files in tmp folder
    tmp_path = FL_BENCH_ROOT / 'tmp'
    for tmp_file in tmp_path.glob('*'):
        tmp_file.unlink()
    
    # Get dataset
    name = cfg.name.lower()
    data, csv = get_dataset(name)
    # print length of unique values for each column
    
    # Prepare dataset
    data = dataset_prepare(name, data)
    
    # Split dataset for each client (iid or niid)
    if cfg.iid:
        num_clients = cfg.num_clients
        # randomly split dataset into num_clients parts
        if csv:
            data = data.sample(frac=1).reset_index(drop=True)
            # split dataset into num_clients parts
            data_split = [data.iloc[i::num_clients, :].reset_index(drop=True) for i in range(num_clients)]
        else:
            raise NotImplementedError(f'Not implemented for non-csv dataset {name}')
    else:
        raise NotImplementedError(f'Not implemented for non-iid dataset {name}')
    
    # Save dataset for each client
    if cfg.federated:
        for i, data in enumerate(data_split):
            data.to_csv(tmp_path / f'data_{i}.csv', index=False)
    else:
        data.to_csv(tmp_path / 'data.csv', index=False)

    # Set input dimension
    cfg.num_features = data.shape[1] - 1

    return cfg


            
        



