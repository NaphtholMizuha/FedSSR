import torch
from loguru import logger
from ..aggregation import aggregate
from ..attack import attack
from typing import TYPE_CHECKING

from .regression import CreditRegressor
from . import score

if TYPE_CHECKING:
    from .base import Experiment

MIN_TEMPERATURE = 0.05
IMPORTANCE_TEMPERATURE = 0.1
NUM_SCORES = 3


class MoziFLHandler:
    def __init__(self, experiment: "Experiment"):
        """
        Initializes the handler for the MoziFL experiment.

        Args:
            experiment (Experiment): The main experiment object.
        """
        self.exp = experiment

        # Instantiate the regressor
        self.regressor = CreditRegressor(n_client=self.exp.n_client)

        # MoziFL specific state
        self.credit = torch.zeros(self.exp.n_client)
        self.prev_winner = 1
        self.sampling_dim = 1000

    def run_round(self, r: int):
        """
        Runs a single round of the MoziFL algorithm.

        Args:
            r (int): The current round number.

        Returns:
            tuple[float, float]: The loss and accuracy of the global model after the round.
        """
        # local training and simulating attacks
        logger.info(f"Round {r}: Start Training")
        for client in self.exp.clients:
            client.local_train(self.exp.n_epoch)
        logger.info(f"Round {r}: Training End.")
        client_updates = torch.stack(
            [client.get_grad() for client in self.exp.clients]
        )
        client_updates = attack(
            client_updates, self.exp.attack, self.exp.m_client, self.exp.n_client
        )

        # New logic: aggregate top frac% clients based on credit for the global update
        num_top_clients = int(self.exp.n_client * self.exp.frac)
        if num_top_clients == 0:
            num_top_clients = 1  # Fallback to 1 client if frac is too small

        _, top_client_indices = torch.topk(self.credit, num_top_clients)

        global_update = aggregate(client_updates[top_client_indices], "fedavg")

        num_mali_in_top = (top_client_indices < self.exp.m_client).sum().item()
        logger.info(
            f"Round {r}: Aggregating top {num_top_clients} clients ({self.exp.frac * 100:.1f}%) with highest credit for global update. "
            f"{num_mali_in_top} are malicious."
        )

        # The rest is for credit model training.
        # It requires frac > 0 for client selection.
        if self.exp.frac > 0:
            # select client sbubsets for credit model training
            selected_index = self._select_clients(
                num_selected=int(self.exp.frac * self.exp.n_client)
            )

            mali = torch.sum(selected_index < self.exp.m_client, dim=1)
            logger.debug(f"Round {r}: Server Selection for credit model:\n {mali}.")

            # aggregate and score
            server_updates = self._get_server_updates(client_updates, selected_index)

            scores = score.calculate_scores(
                client_updates, server_updates, self.exp.score_types
            )
            logger.debug(f"Round scores: {scores}")

            self.regressor.collect_data(scores, selected_index, self.exp.n_server)

            if self.regressor.should_retrain(r):
                logger.info(f"Round {r}: Retraining credit model")
                new_credits = self.regressor.train_and_get_credits()
                if new_credits is not None:
                    self.credit = new_credits

            self._log_credit_stats()

            
        for client in self.exp.clients:
            client.set_grad(global_update)

        # test and log
        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc

    def _select_clients(self, num_selected: int) -> torch.Tensor:
        """
        Selects clients for participation uniformly at random.

        Args:
            num_selected (int): The number of clients to select for each server.

        Returns:
            torch.Tensor: A tensor of selected client indices for each server.
        """
        return torch.stack(
            [
                torch.randperm(self.exp.n_client)[:num_selected]
                for _ in range(self.exp.n_server)
            ]
        )

    def _get_server_updates(
        self, client_updates: torch.Tensor, selected_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates server updates by aggregating updates from selected clients.

        Args:
            client_updates (torch.Tensor): A tensor of all client updates.
            selected_index (torch.Tensor): A tensor of selected client indices for each server.

        Returns:
            torch.Tensor: A tensor of aggregated server updates.
        """
        server_updates = []
        for i in range(self.exp.n_server):
            if i < self.exp.m_server:
                update = aggregate(client_updates, "collude", m=self.exp.m_client)
            else:
                update = aggregate(client_updates[selected_index[i]], "fedavg")
            server_updates.append(update)
        return torch.stack(server_updates)

    def _log_credit_stats(self):
        """
        Logs the average credit for benign and malicious clients.
        """
        benign_avg_credit = self.credit[self.exp.m_client :].mean()
        malicious_avg_credit = self.credit[: self.exp.m_client].mean()
        logger.debug(f"Benign: {benign_avg_credit:.4f}, Mal: {malicious_avg_credit:.4f}")
