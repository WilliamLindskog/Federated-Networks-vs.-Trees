import sys
import json
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from pathlib import Path
from typing import Tuple, Any, Union, Optional
from sklearn.model_selection import train_test_split

from src.benchmark.utils import (
    empty_dir
)
from src.benchmark.constants import PATHS
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

def get_dataset(name: str, femnist_cfg: DictConfig = None) -> pd.DataFrame:
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
    if name == 'heart':
        try:
            data = pd.read_csv(FL_BENCH_ROOT / 'data' / 'heart_disease' / 'heart.csv')
        except FileNotFoundError:
            _heart_data_format(root=FL_BENCH_ROOT)
    elif name in ['femnist', 'synthetic']:
        # Set femnist and data path
        data_root = FL_BENCH_ROOT / Path(PATHS[name])
        data_dir = data_root / 'data'
        
        # remove folders "train", "test", "rem_user_data", "sampled_data" in data_path
        if name == 'femnist':
            assert femnist_cfg is not None, 'Femnist config is None'
            if femnist_cfg.delete_old_partitions:
                _remove_old_femnist_partitions(data_path)
            data_path = data_dir / 'femnist.csv'  
        else:
            data_path = data_dir / 'synthetic.csv'

        # Create femnist df if not exists
        if not data_path.exists():
            _create_df(data_path, data_path, tag=f'{name}')

        # Read femnist data
        data = pd.read_csv(data_path)
    elif name in ['insurance', 'machine', 'lumpy', 'smoking']:
        data = pd.read_csv(FL_BENCH_ROOT / Path(PATHS[name]))
    else:
        raise NotImplementedError(f'Unknown dataset {name}')

    return data

def dataset_main(cfg: DictConfig) -> Optional[DictConfig]:
    """Main function for dataset."""
    # Remove files in tmp folder
    tmp_path = FL_BENCH_ROOT / 'tmp'
    empty_dir(tmp_path)
    
    # Get dataset
    name = cfg.dataset.name.lower()
    data = get_dataset(name) if name != 'femnist' else get_dataset(name, cfg.femnist)
    
    # Prepare dataset
    data = dataset_prepare(name, data, cfg)

    if cfg.dataset.server_dataset:
        data = _set_server_data(cfg, data, tmp_path)
    
    # Split dataset for each client (iid or niid)
    if cfg.dataset.iid:
        num_clients = cfg.dataset.num_clients
        # randomly split dataset into num_clients parts
        data = data.sample(frac=1).reset_index(drop=True)
        # split dataset into num_clients parts
        data_split = [data.iloc[i::num_clients, :].reset_index(drop=True) for i in range(num_clients)]
    else:
        if cfg.dataset.split_by_classes:
            targets_numpy = np.array(data.targets, dtype=np.int32)
            label_list = list(range(len(data.classes)))

            data_idx_for_each_label = [
                np.where(targets_numpy == i)[0].tolist() for i in label_list
            ]

            assigned_labels = []
            selected_times = [0 for _ in label_list]
            for _ in range(client_num):
                sampled_labels = random.sample(label_list, class_num)
                assigned_labels.append(sampled_labels)
                for j in sampled_labels:
                    selected_times[j] += 1

            batch_sizes = _get_batch_sizes(
                targets_numpy=targets_numpy,
                label_list=label_list,
                selected_times=selected_times,
            )

            data_indices = _get_data_indices(
                batch_sizes=batch_sizes,
                data_indices=data_indices,
                data_idx_for_each_label=data_idx_for_each_label,
                assigned_labels=assigned_labels,
                client_num=client_num,
            )

            partition["data_indices"] = data_indices

            return partition 
        else:
            raise NotImplementedError(f'Not implemented for non-csv dataset {name}')    
    # Save dataset for each client
    if cfg.dataset.federated:
        for i, data in enumerate(data_split):
            data.to_csv(tmp_path / f'data_{i}.csv', index=False)
    else:
        data.to_csv(tmp_path / 'data.csv', index=False)

    # Set input dimension
    cfg.dataset.num_features = data.shape[1] - 1

    if cfg.dataset.name == 'synthetic':
        tmp_data = pd.read_csv(f'./data/leaf/data/synthetic/data/synthetic.csv')
        cfg.dataset.num_classes = len(tmp_data['y'].unique())
        cfg.task = 'multiclass'

    return cfg


def _remove_old_femnist_partitions(data_path: Path) -> None:
    """Remove old femnist partitions."""
    for folder in ['train', 'test', 'rem_user_data', 'sampled_data']:
        subprocess.run(['rm', '-rf', data_path / folder])
        
def _create_df(data_path: Path, femnist_data_path: Path, tag: str = None) -> None:
    """ Create femnist df.
    
    Parameters
    ----------
    data_path : Path
        Path to data.
    femnist_data_path : Path
        Path to femnist data.
        
    Returns
    -------
    None
    """
    train_path, test_path = data_path / 'train', data_path / 'test'
    df_list = []
    for _, path in enumerate([train_path, test_path]):
        for file in path.glob('*'):
            print(file)
            with open(file) as f:
                data = json.load(f)
                users = data['users']
                for user in users:
                    user_data = {'x': data['user_data'][user]['x'], 'y': data['user_data'][user]['y']}
                    for i in range(len(user_data['x'])):
                        user_data['x'][i] = np.array(user_data['x'][i])
                        for j in range(len(user_data['x'][i])):
                            user_data[f'x_{j}'] = user_data['x'][i][j]
                        user_data['y'][i] = np.array(user_data['y'][i])
                        df_temp = pd.DataFrame({k: [v] for k, v in user_data.items() if k not in ['x', 'y']})
                        df_temp['y'] = user_data['y'][i]
                        df_temp['user'] = user
                        df_list.append(df_temp)
    # name df based on n
    df = pd.concat(df_list)
    df.to_csv(femnist_data_path, index=False)

def _heart_data_format(root: Path) -> None:
    """ Format heart data. """
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',  
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    data_list = []
    for i, data_file in enumerate((root / 'data' / 'heart_disease').glob('*')):
        data = pd.read_csv(data_file, header=None, names=columns)
        data['region'] = i
        # if value target is larger than 0, set to 1
        data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
        data_list.append(data)
    data = pd.concat(data_list)
    data.to_csv(root / 'data' / 'heart_disease' / 'heart.csv', index=False)

def _set_server_data(cfg: DictConfig, data: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    """Set server data."""
    # split dataset into client and server data
    data, server_data = train_test_split(data, test_size=cfg.dataset.server_dataset_frac, random_state=42)
    server_data.to_csv(tmp_path / 'server_data.csv', index=False)

    return data