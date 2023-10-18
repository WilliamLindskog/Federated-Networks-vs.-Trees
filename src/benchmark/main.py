import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Tuple, Union
from src.benchmark.dataset import dataset_main

@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
    """Main function.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    """
    
    # 1. Print config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare dataset
    dataset_main(cfg.dataset)

    # 3. Prepare model



if __name__ == '__main__':
    main()