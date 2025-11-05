import torch
from loguru import logger
import numpy as np
from ..aggregation import aggregate
from ..attack import attack
from typing import TYPE_CHECKING

from .regression import CreditRegressor
from . import score

if TYPE_CHECKING:
    from .base import Experiment

MIN_TEMPERATURE = 0.05
IMPORTANCE_TEMPERATURE = 1

class FedSSRHandler:
    def __init__(self, experiment: "Experiment"):
        """
        Initializes the handler for the MoziFL experiment.

        Args:
            experiment (Experiment): The main experiment object.
        """
        self.exp = experiment
        self.regressor = CreditRegressor(n_client=self.exp.n_client)
        self.credit = torch.zeros(self.exp.n_client)
        self.init_state = self.exp.clients[0].state.clone()
        self.rollback = True

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

        if self.exp.frac > 0:
            # select client sbubsets for credit model training
            selected_index = self._select_clients(
                num_selected=self.exp.n_client // 2, temperature=1e-1
            )

            mali = torch.tensor([torch.sum(s < self.exp.m_client) for s in selected_index])

            logger.debug(f"Round {r}: Server Selection for credit model:\n {mali}.")

            # aggregate and score
            server_updates = self._get_server_updates(client_updates, selected_index)
            
            if hasattr(self.exp, 'root'):
                root = self.exp.root
                root.local_train(self.exp.n_epoch)
                clean_update = root.get_grad()
                scores = score.calculate_scores(
                    clean_update, server_updates, self.exp.score_types
                )
            else:
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
            
            if torch.all(self.credit == 0):
                logger.info("All client credits are zero, using highest score server for global update.")
                # If multiple score types are used, average them to get a single score per server.
                agg_scores = scores.mean(dim=1) if scores.ndim > 1 else scores
                best_server_idx = torch.argmax(agg_scores)
                logger.debug(f"Round {r}: Selected server {best_server_idx} with highest score {agg_scores[best_server_idx]}.")
                global_update = server_updates[best_server_idx]
            else:
                # Top-1 server selection based on client credits
                server_credits_list = []
                for server_selection in selected_index:
                    server_credits_list.append(self.credit[server_selection].sum())
                server_credits = torch.stack(server_credits_list)

                # Select the server with the highest total credit
                best_server_idx = torch.argmax(server_credits)
                logger.debug(f"Round {r}: Selected server {best_server_idx} with credit {server_credits[best_server_idx]}.")

                # Global update is the update from the best server
                global_update = server_updates[best_server_idx]

        for client in self.exp.clients:
            client.set_grad(global_update)
            
        if hasattr(self.exp, 'root'):
            root.set_grad(global_update)

        # test and log
        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc

    def _get_trusted_clients() -> torch.Tensor:
        """
        Selects trusted clients based on their credit scores.

        Returns:
            torch.Tensor: Indices of clients with credit > 0.
        """
        return torch.where(self.credit >= 0)[0]

    def _select_clients(
        self, num_selected: int, temperature: float = 0.1
    ) -> list:
        """
        Selects clients for participation based on their credit scores using weighted random sampling.
        Clients with higher credit have a higher probability of being selected.
        """
        # Fallback to random selection if all credits are zero
        if torch.all(self.credit == 0):
            logger.info("All client credits are zero, falling back to random selection.")
            all_indices = torch.arange(self.exp.n_client)
            selections = []
            for _ in range(self.exp.n_server):
                perm = torch.randperm(self.exp.n_client)
                selections.append(all_indices[perm[:num_selected]])
            return selections

        # Use softmax on credits to get sampling probabilities.
        # Higher temperature -> more uniform/random selection
        # Lower temperature -> more greedy selection (picking high-credit clients)
        probs = torch.softmax(self.credit / temperature, dim=0)

        selections = []
        for _ in range(self.exp.n_server):
            # Sample `num_selected` clients without replacement based on the probabilities
            selected_indices = torch.multinomial(
                probs, num_samples=num_selected, replacement=False
            )
            selections.append(selected_indices)
        return selections

    def _get_server_updates(
        self, client_updates: torch.Tensor, selected_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates server updates by aggregating updates from selected clients.

        Args:
            client_updates (torch.Tensor): A tensor of all client updates.
            selected_index (list[torch.Tensor]): A list of tensors of selected client indices for each server.

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
        if self.exp.m_client > 0:
            malicious_avg_credit = self.credit[: self.exp.m_client].mean()
            logger.debug(f"Malicious avg credit: {malicious_avg_credit.cpu().numpy()}")
        if self.exp.m_client < self.exp.n_client:
            benign_avg_credit = self.credit[self.exp.m_client :].mean()
            logger.debug(f"Benign avg credit: {benign_avg_credit.cpu().numpy()}")
            
        logger.debug(f"credits: {self.credit.cpu().numpy()}")

