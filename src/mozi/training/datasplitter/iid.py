from .base import DataSplitter
from torch.utils.data import Dataset
import numpy as np

class IidSplitter(DataSplitter):
    def __init__(self, dataset: Dataset, n_client: int, root_data_size: int = 0) -> None:
        super().__init__(dataset, n_client, root_data_size)
    
    def calc_split_map(self):
        """
        Splits self.dataset_for_splitting into n_client IID partitions.
        Returns indices relative to self.dataset_for_splitting.
        """
        # self.dataset_for_splitting is the client pool Subset
        num_items = len(self.dataset_for_splitting)
        idcs = np.arange(num_items)
        np.random.shuffle(idcs)
        
        # np.array_split handles uneven splits gracefully
        return np.array_split(idcs, self.n_client)