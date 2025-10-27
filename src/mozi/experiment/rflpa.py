import torch
from loguru import logger
import torch.nn.functional as F
from ..aggregation import aggregate
from ..attack import attack
from .score import cos_sim_mat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Experiment


class RFLPAHandler:
    def __init__(self, experiment: "Experiment"):
        """
        Initializes the handler for the classic federated learning (baseline) experiment.

        Args:
            experiment (Experiment): The main experiment object.
        """
        self.exp = experiment

    def run_round(self, r: int):

        root = self.exp.root
        logger.info(f"Round {r}: Start Training")
        for client in self.exp.clients:
            client.local_train(self.exp.n_epoch)
        root.local_train(self.exp.n_epoch)
        logger.info(f"Round {r}: Training End.")

        client_updates = torch.stack([client.get_grad() for client in self.exp.clients])
        client_updates = attack(
            client_updates, self.exp.attack, self.exp.m_client, self.exp.n_client
        )
        root_update = root.get_grad()
        root_norm = root_update.norm()
        ts = cos_sim_mat(root_update, client_updates).flatten().clamp(min=0)
        logger.success(f"truth score:\n{ts}")
        logger.info(client_updates.shape)
        clinet_updates = F.normalize(client_updates, dim=1) * root_norm
        grad_g = aggregate(clinet_updates, "weighted", weights=ts)

        

        for client in self.exp.clients:
            client.set_grad(grad_g)
            
        root.set_grad(grad_g)

        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc