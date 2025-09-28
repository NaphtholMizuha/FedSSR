from .base import DataSplitter
from torch.utils.data import Dataset
import numpy as np

class DirichletSplitter(DataSplitter):
    def __init__(self, dataset: Dataset, n_client: int, alpha: float) -> None:
        super().__init__(dataset, n_client)
        self.alpha = alpha
        
    def calc_split_map(self):
        print(f"Splitting data into {self.n_client} clients using Dirichlet distribution with alpha={self.alpha}")
        labels = np.array([item[1] for item in self.dataset])
        n_class = np.max(labels) + 1
        label_dist = np.random.dirichlet([self.alpha] * self.n_client, n_class)
        class_idcs = [
            np.argwhere(labels == y).flatten() for y in range(n_class)
        ]
        client_idcs = [[] for _ in range(self.n_client)]

        for k_idcs, fracs in zip(class_idcs, label_dist):
            for i, idcs in enumerate(
                np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))
            ):
                client_idcs[i] += [idcs]

        return [np.concatenate(idcs) for idcs in client_idcs]
