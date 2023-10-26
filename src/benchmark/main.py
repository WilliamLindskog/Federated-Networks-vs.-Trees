import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Tuple, Union
from src.benchmark.dataset import dataset_main
import pandas as pd

from src.benchmark.utils import (
    get_model, 
    set_model_target, 
    initial_assertions,
    set_dataset_task,
    get_dataset, 
    get_on_fit_config, 
    save_results_as_pickle,
    plot_metric_from_history,
)
from src.benchmark.client import get_client_fn_simulation
from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from src.benchmark.server import get_evaluate_fn

from pathlib import Path

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
    if cfg.dataset.name == 'synthetic':
        tmp_data = pd.read_csv(f'./data/leaf/data/synthetic/data/synthetic.csv')
        cfg.dataset.num_classes = len(tmp_data['y'].unique())
        cfg.task = 'multiclass'

    # 3. Prepare model
    model = get_model(cfg.model)

    # 4. Train model
    if cfg.federated == False:
        train_loader, test_loader = get_dataset(cfg.dataset)
        model.train(train_loader)
        model.evaluate(test_loader)
        quit()
    else: 
        # on_fit_config = get_on_fit_config(cfg)
        client_fn = get_client_fn_simulation(cfg, model)
        evaluate_fn = None
        if cfg.dataset.server_dataset:
            evaluate_fn = get_evaluate_fn(cfg.dataset, model)

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }

    strategy = instantiate(cfg.strategy) if evaluate_fn is None else instantiate(cfg.strategy, evaluate_fn=evaluate_fn)

    history = start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=ServerConfig(num_rounds=cfg.num_rounds),
        client_resources=client_resources,
        strategy=strategy,
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # 6. Save your results
    save_path = Path(HydraConfig.get().runtime.output_dir)

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(
        history,
        file_path=save_path,
    )
    # plot results and include them in the readme
    #strategy_name = strategy.__class__.__name__
    file_suffix: str = (
    #    f"_{strategy_name}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_lr={cfg.learning_rate}"
    )

    plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
        metric_type=cfg.metric_type,
    )

if __name__ == '__main__':
    main()