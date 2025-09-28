from torch.utils.data import Dataset, Subset
from abc import abstractmethod
class DataSplitter():
    def __init__(self, dataset: Dataset, n_client: int) -> None:
        self.dataset = dataset
        self.n_client = n_client
    @abstractmethod
    def calc_split_map(self):
        pass
    
    def create_subset_from_map(self, idcs_li):
        subsets = []
        for idcs in idcs_li:
            subsets.append(Subset(self.dataset, idcs))
        return subsets
    
    def split(self):
        return self.create_subset_from_map(self.calc_split_map())
    
    
