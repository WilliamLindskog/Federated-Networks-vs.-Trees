import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Tuple, Union
from src.benchmark.dataset import dataset_main

from src.benchmark.utils import (
    get_model, 
    set_model_target, 
    initial_assertions,
    set_dataset_task,
    get_centralized_dataset
)

@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
    """Main function.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    """
    # 0. Set model target and complete initial assertions
    cfg = set_model_target(cfg.model.name, cfg)
    cfg = set_dataset_task(cfg.dataset.name, cfg)

    initial_assertions(cfg)

    # 1. Print config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare dataset
    cfg.dataset = dataset_main(cfg.dataset)

    # 3. Prepare model
    model = get_model(cfg.model)

    # 4. Train model
    if cfg.federated == False:
        train_loader, test_loader = get_centralized_dataset(cfg.dataset)
        model.train(train_loader)
        model.evaluate(test_loader)
    else: 
        pass

if __name__ == '__main__':
    main()