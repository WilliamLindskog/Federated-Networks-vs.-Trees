from omegaconf import DictConfig, OmegaConf

def get_config(path: str) -> DictConfig:
    """Get config from yaml file"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    return cfg 