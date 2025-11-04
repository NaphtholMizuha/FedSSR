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
    # --- CHANGED: Added beta parameter for momentum ---
    def __init__(self, experiment: "Experiment", beta: float = 0.9):
        """
        Initializes the handler for the FedSSR experiment.

        Args:
            experiment (Experiment): The main experiment object.
            beta (float): The momentum coefficient for the server momentum.
        """
        self.exp = experiment
        self.regressor = CreditRegressor(n_client=self.exp.n_client)
        self.credit = torch.zeros(self.exp.n_client)
        self.init_state = self.exp.clients[0].state.clone()
        self.rollback = True
        
        # --- ADDED: Server momentum state ---
        self.beta = beta
        # Initialize to None. It will be set with the first global_update.
        # This makes the logic for the first round simple and clean.
        self.server_momentum = None 
        
        # --- CHANGED: Renamed for clarity, and set to True by default ---
        # This flag now controls whether to use server momentum for scoring.
        self.use_momentum_scoring = False 
        
        # --- REMOVED: No longer needed ---
        # self.prev_global_update = None
        # self.use_prev_global_update_scoring = False


    def run_round(self, r: int):
        """
        Runs a single round of the FedSSR algorithm.
        ...
        """
        # (The first part of your function remains exactly the same)
        # ... local training, attack simulation ...
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
            # ... client selection logic remains the same ...

            num_selected = (self.credit >= 0).sum().item()
            if num_selected in [0, self.exp.n_client]:
                num_selected = int(self.exp.frac * self.exp.n_client)
            selected_index = self._select_clients(
                num_selected=int(self.exp.frac * self.exp.n_client), temperature=0.1
            )
            mali = torch.sum(selected_index < self.exp.m_client, dim=1)
            logger.debug(f"Round {r}: Server Selection for credit model:\n {mali}/{num_selected}.")

            server_updates = self._get_server_updates(client_updates, selected_index)
            
            # --- THIS IS THE MAIN LOGIC CHANGE FOR SCORING ---
            if self.use_momentum_scoring:
                logger.info("Using server momentum for scoring.")
                # Use the momentum vector as the reference
                reference_update = self.server_momentum
                # The existing check for the first round (when reference is None) works perfectly!
                if reference_update is None:
                    logger.warning("Momentum is not yet initialized (first round). Using zero vector as reference.")
                    reference_update = torch.zeros_like(server_updates[0])
                scores = score.calculate_scores(
                    reference_update, server_updates, self.exp.score_types
                )
            # The fallback logic remains the same
            elif hasattr(self.exp, 'root'):
                logger.info("Using root client for scoring.")
                root = self.exp.root
                root.local_train(self.exp.n_epoch)
                clean_update = root.get_grad()
                scores = score.calculate_scores(
                    clean_update, server_updates, self.exp.score_types
                )
            else:
                logger.info("Using client updates for scoring.")
                scores = score.calculate_scores(
                    client_updates, server_updates, self.exp.score_types
                )
            # --- END OF SCORING LOGIC CHANGE ---

            logger.debug(f"Round scores: {scores}")

            self.regressor.collect_data(scores, selected_index, self.exp.n_server)

            if self.regressor.should_retrain(r):
                logger.info(f"Round {r}: Retraining credit model")
                # --- CHANGED: Switched the flag name for clarity ---
                if not hasattr(self.exp, 'root'):
                    logger.info("Switching to use server momentum for scoring.")
                    self.use_momentum_scoring = True
                new_credits = self.regressor.train_and_get_credits()
                if new_credits is not None:
                    self.credit = new_credits

            self._log_credit_stats()
            
        winner = scores.argmax()
        
        # ... winner checking logic remains the same ...
        if mali[winner] > mali.min():
            logger.warning(
                f"Round {r}: Winner {winner.item()} has {mali[winner]} malicious clients, "
                f"while the best server has {mali.min()}."
            )
            
        logger.debug(f"Round {r}: Welcome our new winner: {winner.item()}!")
        global_update = server_updates[winner]
        
        # --- ADDED: Update the server momentum after selecting the winner ---
        if self.server_momentum is None:
            # First round: initialize momentum with the first global update
            self.server_momentum = global_update.clone()
        else:
            # Subsequent rounds: apply the momentum update rule
            self.server_momentum = self.beta * self.server_momentum + global_update
        
        # --- REMOVED ---
        # self.prev_global_update = global_update

        # The rest of the function remains the same
        for client in self.exp.clients:
            client.set_grad(global_update)
            
        if hasattr(self.exp, 'root'):
            self.exp.root.set_grad(global_update)

        loss, acc = self.exp.clients[-1].test()
        logger.success(f"Round {r}: Loss: {loss:.4f}, Acc: {acc * 100:.2f}!")
        return loss, acc

    # The helper methods (_select_clients, _get_server_updates, etc.) remain unchanged.
    # ...
    # ... (paste your unchanged helper methods here) ...
    # ...
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
                for i in range(self.exp.n_server)
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