import torch
from loguru import logger
from ..aggregation import aggregate
from ..attack import attack
from ..training.parallel_trainer import create_parallel_trainer
from .score import cos_sim_mat

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Experiment


class BSRFLHandler:
    def __init__(self, experiment: "Experiment"):
        """
        Initializes the handler for the BSRFL experiment with parallel training.

        Args:
            experiment (Experiment): The main experiment object.
        """
        self.exp = experiment
        self.weight_base = torch.Tensor()
        # Initialize parallel trainer
        self.parallel_trainer = create_parallel_trainer(self.exp.clients, mode="batch")

    def run_round(self, r: int):
        """
        Runs a single round of BSRFL with parallel client training.

        Args:
            r (int): The current round number.

        Returns:
            tuple[float, float]: The loss and accuracy of the global model after the round.
        """
        if r == 0:
            self.weight_base = self.exp.clients[-1].get_weight()
            
        logger.info(f"Round {r}: Start Parallel Training")
        weights_prev = torch.stack([client.get_weight() for client in self.exp.clients])
        
        # Use parallel training instead of sequential
        self.parallel_trainer.parallel_local_train(self.exp.n_epoch)
        
        logger.info(f"Round {r}: Training End.")

        client_updates = torch.stack([client.get_grad() for client in self.exp.clients])
        client_updates = attack(
            client_updates, self.exp.attack, self.exp.m_client, self.exp.n_client
        )
        
        weights_now = weights_prev + client_updates

        cs = cos_sim_mat(self.weight_base, weights_now).flatten() / 2 + 0.5
        logger.info(f"scores:\n{cs}")
        scores = torch.sigmoid(50 * (cs - 0.5))
        logger.info(f"weights:\n{scores}")
        weight_g = aggregate(weights_now, "weighted", weights=scores)
        logger.info(f"Round {r}: Aggregation End")
        
        scores_base = (cs > 0.5) * scores

        for client in self.exp.clients:
            client.set_weight(weight_g)
            
        self.weight_base = aggregate(weights_now, "weighted", weights=scores_base)

        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc