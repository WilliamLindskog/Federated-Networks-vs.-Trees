import sys
from omegaconf import DictConfig
from pathlib import Path

import pandas as pd

from typing import Tuple, Any, Union, Optional
from src.benchmark.dataset_preparation import dataset_prepare

from sklearn.model_selection import train_test_split
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

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

def get_dataset(name: str, femnist_cfg: DictConfig = None) -> Tuple[Any, bool]:
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
    elif name == 'heart':
        try:
            data = pd.read_csv(FL_BENCH_ROOT / 'data' / 'heart_disease' / 'heart.csv')
        except FileNotFoundError:
            columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',  
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]
            data_list = []
            for i, data_file in enumerate((FL_BENCH_ROOT / 'data' / 'heart_disease').glob('*')):
                data = pd.read_csv(data_file, header=None, names=columns)
                data['region'] = i
                # if value target is larger than 0, set to 1
                data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
                data_list.append(data)
            data = pd.concat(data_list)
            data.to_csv(FL_BENCH_ROOT / 'data' / 'heart_disease' / 'heart.csv', index=False)
        csv = True
    elif name == 'lumpy':
        data = pd.read_csv(FL_BENCH_ROOT / 'data' / 'lumpy_skin' / 'lumpy_skin.csv')
        csv = True
    elif name == 'machine':
        data = pd.read_csv(
            FL_BENCH_ROOT / 'data' / 'machine_maintenance' / 'predictive_maintenance.csv'
        )
        csv = True
    elif name == 'femnist':
        assert femnist_cfg is not None, 'Femnist config is None'
        femnist_path = FL_BENCH_ROOT / 'data' / 'leaf' / 'data' / 'femnist'
        data_path = femnist_path / 'data'
        # remove folders "train", "test", "rem_user_data", "sampled_data" in data_path
        if femnist_cfg.delete_old_partitions:
            for folder in data_path.glob('*'):
                if folder.is_dir() and folder.name in [
                    'rem_user_data', 'sampled_data', 'train', 'test'
                ]:
                    for file in folder.glob('*'):
                        file.unlink()
                    folder.rmdir()
        
        # go thorugh files in data_path train and test
        else:
            femnist_data_path = data_path / 'femnist.csv'
            if not femnist_data_path.exists():
                train_path = data_path / 'train'
                test_path = data_path / 'test'
                df_list = []
                for n, path in enumerate([train_path, test_path]):
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
            csv = True
            data = pd.read_csv(femnist_data_path)
    
    elif name == 'synthetic':
        synthetic_path = FL_BENCH_ROOT / 'data' / 'leaf' / 'data' / 'synthetic'
        data_path = synthetic_path / 'data'

        train_path = data_path / 'train'
        test_path = data_path / 'test'
        df_list = []
        synthetic_data_path = data_path / 'synthetic.csv'
        if not synthetic_data_path.exists():
            for n, path in enumerate([train_path, test_path]):
                for file in path.glob('*'):
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
                df.to_csv(synthetic_data_path, index=False)
        csv = True
        data = pd.read_csv(synthetic_data_path)
    elif name == 'insurance':
        data = pd.read_csv(FL_BENCH_ROOT / 'data' / 'health_insurance' / 'insurance.csv')
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
    data, csv = get_dataset(name) if name != 'femnist' else get_dataset(name, cfg.femnist)
    
    # Prepare dataset
    data = dataset_prepare(name, data)

    if cfg.iid and name == 'femnist':
        data.drop(columns=['user'], inplace=True)

    if cfg.server_dataset:
        # split dataset into client and server data
        data, server_data = train_test_split(data, test_size=cfg.server_dataset_frac, random_state=42)
        server_data.to_csv(tmp_path / 'server_data.csv', index=False)
    
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


            
        



