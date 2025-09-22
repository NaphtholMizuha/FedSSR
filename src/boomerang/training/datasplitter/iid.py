from .base import DataSplitter
from torch.utils.data import Dataset
import numpy as np

class IidSplitter(DataSplitter):
    def __init__(self, dataset: Dataset, n_client: int) -> None:
        super().__init__(dataset, n_client)
    
    def calc_split_map(self):
        idcs = np.arange(len(self.dataset))
        np.random.shuffle(idcs)
        return np.array_split(idcs, self.n_client)
    