from .boomerang.experiment import Experiment, ExperimentConfig
import toml
from loguru import logger

if __name__ == '__main__':
    config = ExperimentConfig(**toml.load('conf/default.toml'))
    logger.info(config)
    experiment = Experiment(config)
    experiment.run()