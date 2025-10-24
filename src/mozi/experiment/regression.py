import numpy as np
import torch
from sklearn.linear_model import Ridge
from loguru import logger
from math import ceil


class CreditRegressor:
    def __init__(
        self, n_client: int, max_history_size: int = 500, retrain_interval: int = 1
    ):
        """
        Initializes the credit regressor.

        Args:
            n_client (int): The total number of clients.
            max_history_size (int): The maximum number of data points to store.
            retrain_interval (int): The interval (in rounds) at which to retrain the model.
        """
        self.n_client = n_client
        self.max_history_size = max_history_size
        self.retrain_interval = retrain_interval

        self.history = []
        self.model = Ridge(alpha=1.0)

    def collect_data(
        self, scores: torch.Tensor, selected_index: torch.Tensor, n_server: int
    ):
        """
        Collects training data for the regression model from a round's results.

        Args:
            scores (torch.Tensor): The scores of the server updates.
            selected_index (torch.Tensor): The indices of clients selected by each server.
            n_server (int): The number of servers.
        """
        new_data_points = []
        for k in range(n_server):
            participation_vector = np.zeros(self.n_client)
            selected_clients = selected_index[k]
            participation_vector[selected_clients.cpu().numpy()] = 1
            target_vector = scores[k].cpu().numpy()
            new_data_points.append((participation_vector, target_vector))

        self.history.extend(new_data_points)
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size :]

    def should_retrain(self, current_round: int) -> bool:
        """
        Determines whether the regression model should be retrained.

        Args:
            current_round (int): The current round number.

        Returns:
            bool: True if the model should be retrained, False otherwise.
        """
        return (
            current_round > 0
            and current_round % self.retrain_interval == 0
            and len(self.history) > self.n_client
        )

    def train_and_get_credits(self) -> torch.Tensor | None:
        """
        Trains the Ridge regression model and returns the updated client credits.

        Returns:
            torch.Tensor | None: A tensor of new client credits, or None if training fails.
        """
        if not self.history or len(self.history) < self.n_client:
            logger.warning("Not enough history for regression. Skipping credit update.")
            return None

        X_train = np.array([item[0] for item in self.history])
        y_train = np.array([item[1] for item in self.history])

        if np.isnan(y_train).any():
            logger.warning("NaNs found in y_train, filling with 0.")
            y_train = np.nan_to_num(y_train)

        sample_importance = np.linalg.norm(y_train, axis=1) + 0.1
        try:
            self.model.fit(X_train, y_train, sample_weight=sample_importance)
            # For multi-target, model.coef_ is (n_targets, n_features)
            new_credits_matrix = torch.from_numpy(self.model.coef_).float().T
            logger.success("Multi-target credit model retrained. Credits updated.")
            logger.info(f"New credit matrix shape: {new_credits_matrix.shape}")
            return new_credits_matrix
        except Exception as e:
            logger.error(f"Failed to train credit model: {e}")
            logger.error(
                f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}"
            )
            return None