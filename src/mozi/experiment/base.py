from ..training import Trainer, fetch_dataset, fetch_datasplitter, fetch_model
from .classic import ClassicFLHandler
from .fedssr import FedSSRHandler
from .bsrfl import BSRFLHandler
from .rflpa import RFLPAHandler
from dataclasses import dataclass
from loguru import logger
from typing import Literal, List
import polars as pl
from torch.utils.data import random_split, Subset
from datetime import datetime
import os


@dataclass
class ExperimentConfig:
    exp_name: str
    model: str
    dataset: str
    split: str
    learning_rate: float
    n_client: int
    m_client: int
    n_server: int
    m_server: int
    device: str
    datapath: str
    batch_size: int
    n_epoch: int
    n_round: int
    num_workers: int
    attack: str
    aggregation: str
    selection_fraction: float
    method: Literal["ours", "baseline"]
    dir_alpha: float | None = None
    score_types: list[str] | None = None
    clean_data: bool = False


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initializes the experiment environment, setting up datasets, models, and handlers.

        Args:
            config (ExperimentConfig): The configuration object for the experiment.
        """
        train_set, self.test_set = fetch_dataset(config.datapath, config.dataset)

 
        if len(train_set) < config.clean_data:
            raise ValueError("Train set has less than 100 samples, cannot create clean_set.")
            
            
        self.root_subset, self.train_subsets = fetch_datasplitter(
            train_set, config.split, config.n_client, config.clean_data, alpha=config.dir_alpha
        ).split()


        self.n_epoch = config.n_epoch
        self.n_round = config.n_round
        self.attack = config.attack
        self.m_client = config.m_client
        self.n_client = config.n_client
        self.n_server = config.n_server
        self.m_server = config.m_server
        self.frac = config.selection_fraction
        self.method = config.method
        self.aggregation = config.aggregation
        self.score_types = config.score_types
        self.config = config
        self.exp_name = config.exp_name

        self.log_dir = "log"
        os.makedirs(self.log_dir, exist_ok=True)
        self.output_file = os.path.join(self.log_dir, f"{self.exp_name}.parquet")

        if os.path.exists(self.output_file):
            logger.warning(
                f"Output file {self.output_file} already exists. Removing it for a fresh start."
            )
            os.remove(self.output_file)

        self.reset(config)

        # Instantiate the correct handler
        if config.method == "baseline":
            self.handler = ClassicFLHandler(self)
        elif config.method == "ours":
            self.handler = FedSSRHandler(self)
        elif config.method == "bsrfl":
            self.handler = BSRFLHandler(self)
        elif config.method == "rflpa":
            self.handler = RFLPAHandler(self)
        else:
            raise ValueError(f"Unknown method: {config.method}")

    def reset(self, config: ExperimentConfig):
        """
        Resets the experiment state, including models and client trainers.

        Args:
            config (ExperimentConfig): The configuration to reset the experiment with.
        """
        init_model = fetch_model(config.model)
        self.local_models = [
            fetch_model(config.model).to(config.device) for _ in range(config.n_client)
        ]
        self.clients = [
            Trainer(
                model=self.local_models[i],
                init_state=init_model.state_dict(),
                train_set=self.train_subsets[i],
                test_set=self.test_set,
                bs=config.batch_size,
                nw=config.num_workers,
                lr=config.learning_rate,
                device=config.device,
                total_epoch=config.n_epoch*config.n_round,
            )
            for i in range(config.n_client)
        ]
        
        if config.clean_data != 0:
            self.root = Trainer(
                model=fetch_model(config.model).to(config.device),
                init_state=init_model.state_dict(),
                train_set=self.root_subset,
                test_set=self.test_set,
                bs=config.batch_size,
                nw=config.num_workers,
                lr=config.learning_rate,
                device=config.device,
                total_epoch=config.n_epoch*config.n_round,
            )

    def save_results(self, result: dict):
        """
        Saves a result record to the output Parquet file, appending if the file exists.

        Args:
            result (dict): A dictionary containing the round's result data.
        """
        results_df = pl.DataFrame(result)
        try:
            if os.path.exists(self.output_file):
                existing_df = pl.read_parquet(self.output_file)
                combined_df = pl.concat([existing_df, results_df], how="vertical")
                combined_df.write_parquet(self.output_file)
                logger.trace(f"ppended 1 record to {self.output_file}")
            else:
                results_df.write_parquet(self.output_file)
                logger.trace(f"Saved initial record to {self.output_file}")
        except Exception as e:
            logger.error(
                f"❌ Failed to save results to Parquet file {self.output_file}: {e}"
            )

    def run(self):
        """
        Runs the federated learning experiment loop for the configured number of rounds.
        """
        for r in range(self.n_round):
            loss, acc = self.handler.run_round(r)

            record = {
                "exp_name": self.exp_name,
                "timestamp": datetime.now().isoformat(),
                "loss": loss,
                "acc": acc,
                "rnd": r,
            }
            self.save_results(record)