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
IMPORTANCE_TEMPERATURE = 0.1

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
        self.prev_global_update = None
        self.regressor_trained = False
        self.global_updates = []

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

        trusted_client_num = self._get_trusted_clients().numel()

        # global_update = aggregate(client_updates[trusted_client_indices], "fedavg")

        # num_mali_in_trusted = (trusted_client_indices < self.exp.m_client).sum().item()
        # logger.info(
        #     f"Round {r}: Aggregating {len(trusted_client_indices)} clients from the trusted cluster. "
        #     f"{num_mali_in_trusted} are malicious."
        # )

        # The rest is for credit model training.
        # It requires frac > 0 for client selection.
        if self.exp.frac > 0:
            # select client sbubsets for credit model training
            num_selected = int(self.exp.frac * self.exp.n_client)
            if trusted_client_num != self.exp.n_client:
                num_selected = trusted_client_num
            if num_selected == 0:
                num_selected = 1
            selected_index = self._select_clients(
                num_selected=int(self.exp.frac * self.exp.n_client), temperature=0.1
            )

            mali = torch.sum(selected_index < self.exp.m_client, dim=1)

            logger.debug(f"Round {r}: Server Selection for credit model:\n {mali}.")

            # aggregate and score
            server_updates = self._get_server_updates(client_updates, selected_index)
            
            if self.regressor_trained:
                logger.info("Using predicted global update for scoring.")
                if len(self.global_updates) > 1:
                    reference_update = self._predict_next_update(self.global_updates)
                else:
                    reference_update = self.prev_global_update
                if reference_update is None:
                    reference_update = torch.zeros_like(server_updates[0])
                scores = score.calculate_scores(
                    reference_update.to(server_updates.device),
                    server_updates,
                    self.exp.score_types,
                )
            else:
                logger.info("Using client updates for scoring.")
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
                    self.regressor_trained = True


            self._log_credit_stats()
            
        winner = scores.argmax()

        # Check if the winner is the best choice
        if mali[winner] > mali.min():
            logger.warning(
                f"Round {r}: Winner {winner.item()} has {mali[winner]} malicious clients, "
                f"while the best server has {mali.min()}."
            )

        logger.debug(f"Round {r}: Welcome our new winner: {winner.item()}!")
        global_update = server_updates[winner]
        self.prev_global_update = global_update
        self.global_updates.append(global_update.cpu().clone())

        for client in self.exp.clients:
            client.set_grad(global_update)
            
        if hasattr(self.exp, 'root'):
            root.set_grad(global_update)

        # test and log
        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc

    def _predict_next_update(self, updates: list[torch.Tensor], k: int = 2) -> torch.Tensor:
        if len(updates) < k:
            return updates[-1]

        weights = torch.nn.Parameter(torch.randn(k))
        optimizer = torch.optim.LBFGS([weights], lr=0.1, max_iter=20)
        history = torch.stack(updates)

        def closure():
            optimizer.zero_grad()
            loss = 0
            for i in range(k, len(history)):
                target = history[i]
                features = history[i - k : i].flip(dims=[0])

                pred = torch.einsum("i,ij->j", weights, features)

                loss = loss + torch.nn.functional.mse_loss(pred, target)

            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            features = torch.stack(updates[-k:]).flip(dims=[0])
            pred_next = torch.einsum("i,ij->j", weights, features)

        return pred_next

    def _get_trusted_clients(self) -> torch.Tensor:
        """
        Selects trusted clients based on their credit scores.

        Returns:
            torch.Tensor: Indices of clients with credit > 0.
        """
        return torch.where(self.credit >= 0)[0]

    def _select_clients(
        self, num_selected: int, temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Selects clients for participation based on their credits using multinomial sampling.

        Args:
            num_selected (int): The number of clients to select for each server.
            temperature (float): The temperature for the softmax function.

        Returns:
            torch.Tensor: A tensor of selected client indices for each server.
        """
        return torch.stack(
            [
                torch.multinomial(
                    torch.softmax(self.credit / temperature, dim=0).cpu() ,
                    num_samples=num_selected,
                    replacement=False,
                )
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

