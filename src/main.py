from .mozi.experiment import Experiment, ExperimentConfig
import toml
from loguru import logger
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the FedMozi experiment.")
    parser.add_argument(
        '--config',
        '-c',
        type=str, 
        default='conf/default.toml', 
        help='Path to the configuration file (default: conf/default.toml)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    exp_name = args.config.split('/')[-1].replace('.toml', '')
    config = ExperimentConfig(exp_name=exp_name, **toml.load(args.config))
    logger.info(config)
    experiment = Experiment(config)
    experiment.run()