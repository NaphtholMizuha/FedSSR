import torch
from loguru import logger
from ..aggregation import aggregate
from ..attack import attack
from ..training.parallel_trainer import create_parallel_trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Experiment


class ClassicFLHandler:
    def __init__(self, experiment: "Experiment"):
        """
        Initializes the handler for the classic federated learning (baseline) experiment.

        Args:
            experiment (Experiment): The main experiment object.
        """
        self.exp = experiment
        # Initialize parallel trainer
        self.parallel_trainer = create_parallel_trainer(self.exp.clients, mode="batch")

    def run_round(self, r: int):
        """
        Runs a single round of classic federated learning with parallel client training.

        Args:
            r (int): The current round number.

        Returns:
            tuple[float, float]: The loss and accuracy of the global model after the round.
        """
        logger.info(f"Round {r}: Start Parallel Training")
        
        # Use parallel training instead of sequential
        self.parallel_trainer.parallel_local_train(self.exp.n_epoch)
        
        logger.info(f"Round {r}: Training End.")

        client_updates = torch.stack([client.get_grad() for client in self.exp.clients])
        client_updates = attack(
            client_updates, self.exp.attack, self.exp.m_client, self.exp.n_client
        )

        server_updates = []
        for i in range(self.exp.n_server):
            if i < self.exp.m_server:
                # when the server is malicious
                update = aggregate(client_updates, "collude", m=self.exp.m_client)
            else:
                # when the server is benign
                update = aggregate(client_updates, "fedavg")
            server_updates.append(update)
        server_updates = torch.stack(server_updates)

        logger.info(f"Round {r}: Aggregation End")
        global_update = aggregate(server_updates, self.exp.aggregation, prop=0.8)

        for client in self.exp.clients:
            client.set_grad(global_update)

        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc