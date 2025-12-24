import torch
from loguru import logger
import torch.nn.functional as F
from ..aggregation import aggregate
from ..attack import attack
from ..training.parallel_trainer import create_parallel_trainer
import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Experiment


class RFedHandler:
    def __init__(self, experiment: "Experiment"):
        """
        Initializes the handler for the RFed experiment with parallel training.

        Args:
            experiment (Experiment): The main experiment object.
        """
        self.exp = experiment
        # Initialize parallel trainer
        self.parallel_trainer = create_parallel_trainer(self.exp.clients, mode="batch")

    def run_round(self, r: int):
        """
        Runs a single round of RFed with parallel client training.

        Args:
            r (int): The current round number.

        Returns:
            tuple[float, float]: The loss and accuracy of the global model after the round.
        """
        root = self.exp.root
        logger.info(f"Round {r}: Start Parallel Training")
        
        # Use parallel training for clients
        self.parallel_trainer.parallel_local_train(self.exp.n_epoch)
        
        # Train root model separately (it's typically small)
        root.local_train(self.exp.n_epoch)
        
        logger.info(f"Round {r}: Training End.")

        client_updates = torch.stack([client.get_grad() for client in self.exp.clients])
        client_updates = attack(
            client_updates, self.exp.attack, self.exp.m_client, self.exp.n_client
        )
        root_update = root.get_grad()

        mask = torch.rand(client_updates.shape[1]).to(client_updates.device)
        logger.info(f'mask: {mask.shape}')
        masked_client_updates = client_updates + mask
        scores = masked_client_updates.matmul(root_update) / math.sqrt(float(root_update.numel()))
        weights = scores.softmax(0)
        logger.info(f'weights:\n{weights}')
        grad_g = aggregate(client_updates, "weighted", weights=weights)  

        for client in self.exp.clients:
            client.set_grad(grad_g)
            
        root.set_grad(grad_g)

        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc