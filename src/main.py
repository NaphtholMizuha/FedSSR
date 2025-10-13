from .mozi.experiment import Experiment, ExperimentConfig
import toml
from loguru import logger
import typer
from typing import Annotated, Optional

app = typer.Typer()

@app.command()
def main(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to the configuration file")] = "conf/noattack.toml",
    # task
    model: Annotated[Optional[str], typer.Option(help="Model architecture")] = None,
    dataset: Annotated[Optional[str], typer.Option(help="Dataset name")] = None,
    split: Annotated[Optional[str], typer.Option(help="Data splitting strategy")] = None,
    dir_alpha: Annotated[Optional[float], typer.Option(help="Dirichlet alpha parameter")] = None,
    
    # system
    n_client: Annotated[Optional[int], typer.Option(help="Number of clients")] = None,
    m_client: Annotated[Optional[int], typer.Option(help="Number of malicious clients")] = None,
    n_server: Annotated[Optional[int], typer.Option(help="Number of servers")] = None,
    m_server: Annotated[Optional[int], typer.Option(help="Number of malicious servers")] = None,
    n_round: Annotated[Optional[int], typer.Option(help="Number of rounds")] = None,
    
    # training
    n_epoch: Annotated[Optional[int], typer.Option(help="Number of epochs per round")] = None,
    learning_rate: Annotated[Optional[float], typer.Option(help="Learning rate")] = None,
    device: Annotated[Optional[str], typer.Option(help="Device to use (cuda/cpu)")] = None,
    datapath: Annotated[Optional[str], typer.Option(help="Path to data directory")] = None,
    batch_size: Annotated[Optional[int], typer.Option(help="Batch size")] = None,
    num_workers: Annotated[Optional[int], typer.Option(help="Number of data loader workers")] = None,
    
    # security
    attack: Annotated[Optional[str], typer.Option(help="Attack type")] = None,
    aggregation: Annotated[Optional[str], typer.Option(help="Aggregation method")] = None,
    selection_fraction: Annotated[Optional[float], typer.Option(help="Selection fraction")] = None,
    method: Annotated[Optional[str], typer.Option(help="Method (stateful/stateless/baseline)")] = None,
):
    """
    Run the FedMozi experiment with optional parameter overrides.
    """
    # Load config from file
    config_data = toml.load(config)
    exp_name = config.split('/')[-1].replace('.toml', '')
    
    # Override config with command line arguments if provided
    # task
    if model is not None:
        config_data['model'] = model
    if dataset is not None:
        config_data['dataset'] = dataset
    if split is not None:
        config_data['split'] = split
    if dir_alpha is not None:
        config_data['dir_alpha'] = dir_alpha
    
    # system
    if n_client is not None:
        config_data['n_client'] = n_client
    if m_client is not None:
        config_data['m_client'] = m_client
    if n_server is not None:
        config_data['n_server'] = n_server
    if m_server is not None:
        config_data['m_server'] = m_server
    if n_round is not None:
        config_data['n_round'] = n_round
    
    # training
    if n_epoch is not None:
        config_data['n_epoch'] = n_epoch
    if learning_rate is not None:
        config_data['learning_rate'] = learning_rate
    if device is not None:
        config_data['device'] = device
    if datapath is not None:
        config_data['datapath'] = datapath
    if batch_size is not None:
        config_data['batch_size'] = batch_size
    if num_workers is not None:
        config_data['num_workers'] = num_workers
    
    # security
    if attack is not None:
        config_data['attack'] = attack
    if aggregation is not None:
        config_data['aggregation'] = aggregation
    if selection_fraction is not None:
        config_data['selection_fraction'] = selection_fraction
    if method is not None:
        config_data['method'] = method
    
    config_obj = ExperimentConfig(exp_name=exp_name, **config_data)
    logger.info(config_obj)
    experiment = Experiment(config_obj)
    experiment.run()

if __name__ == "__main__":
    app()