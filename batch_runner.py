
import typer
from typing import Annotated, Optional, List
import os
import glob
import subprocess
import shlex
from loguru import logger

app = typer.Typer()

def build_command_args(
    model: Optional[str], dataset: Optional[str], split: Optional[str], dir_alpha: Optional[float],
    n_client: Optional[int], m_client: Optional[int], n_server: Optional[int], m_server: Optional[int],
    n_round: Optional[int], n_epoch: Optional[int], learning_rate: Optional[float], device: Optional[str],
    datapath: Optional[str], batch_size: Optional[int], num_workers: Optional[int], attack: Optional[str],
    aggregation: Optional[str], selection_fraction: Optional[float], method: Optional[str],
    score_types: Optional[List[str]]
) -> List[str]:
    """Builds a list of command-line arguments from the given parameters."""
    args = {
        "--model": model, "--dataset": dataset, "--split": split, "--dir_alpha": dir_alpha,
        "--n-client": n_client, "--m-client": m_client, "--n-server": n_server, "--m-server": m_server,
        "--n-round": n_round, "--n-epoch": n_epoch, "--learning-rate": learning_rate, "--device": device,
        "--datapath": datapath, "--batch-size": batch_size, "--num-workers": num_workers,
        "--attack": attack, "--aggregation": aggregation, "--selection-fraction": selection_fraction,
        "--method": method, "--score-types": score_types
    }
    
    cmd_args = []
    for key, value in args.items():
        if value is not None:
            if isinstance(value, list):
                for item in value:
                    cmd_args.extend([key, str(item)])
            else:
                cmd_args.extend([key, str(value)])
    return cmd_args

@app.command()
def main(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to a single configuration file.")] = None,
    batch_mode: Annotated[bool, typer.Option("--batch", "-b", help="Enable batch mode to run for all configs in `conf` folder.")] = False,
    
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
    method: Annotated[Optional[str], typer.Option(help="Method (ours/baseline)")] = None,
    score_types: Annotated[Optional[List[str]], typer.Option(help="List of score types to use")] = None,
):
    """
    Run FedMozi experiments, either for a single config or in batch mode for multiple configs.
    Accepts all arguments from src.main.py to override config files.
    """
    if not batch_mode and config is None:
        logger.error("Either --batch mode must be enabled or a --config file must be provided.")
        raise typer.Exit(code=1)
        
    if batch_mode and config is not None:
        logger.warning(f"Both --batch and --config are provided. --config ('{config}') will be ignored.")

    # Ensure log directory exists
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    if batch_mode:
        config_files = glob.glob("conf/*.toml")
        if not config_files:
            logger.warning("Batch mode is on, but no .toml files found in 'conf/' directory.")
            return
    else:
        config_files = [config]

    extra_args = build_command_args(
        model, dataset, split, dir_alpha, n_client, m_client, n_server, m_server,
        n_round, n_epoch, learning_rate, device, datapath, batch_size, num_workers,
        attack, aggregation, selection_fraction, method, score_types
    )

    for conf_file in config_files:
        if not os.path.isfile(conf_file):
            logger.error(f"Config file not found: {conf_file}")
            continue

        session_name = os.path.basename(conf_file).replace('.toml', '')
        log_file = os.path.join(log_dir, f"{session_name}.log")

        # Clear old log file
        if os.path.exists(log_file):
            logger.info(f"Clearing old log file: {log_file}")
            os.remove(log_file)

        # Kill existing tmux session
        if subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True).returncode == 0:
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            logger.info(f"Existing tmux session '{session_name}' killed.")

        # Build and run the command in a new tmux session
        base_command = ["uv", "run", "-m", "src.main", "-c", conf_file]
        full_command = base_command + extra_args
        
        # Use shlex.join for robust command string formatting
        command_str = shlex.join(full_command)
        
        tmux_command = ["tmux", "new-session", "-d", "-s", session_name, command_str]
        
        subprocess.run(tmux_command, check=True)
        logger.info(f"Tmux session '{session_name}' created for config '{conf_file}'.")
        logger.info(f"Command: {command_str}")

if __name__ == "__main__":
    app()
