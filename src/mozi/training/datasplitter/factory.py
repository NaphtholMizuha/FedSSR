from .iid import IidSplitter
from .dirichlet import DirichletSplitter
from .base import DataSplitter
from torch.utils.data import Dataset

def fetch_datasplitter(dataset: Dataset, type: str, n_client: int, root_data_size: int, **kwargs) -> DataSplitter:
    if type == 'iid':
        return IidSplitter(dataset, n_client, root_data_size)
    elif type == 'dir':
        return DirichletSplitter(dataset, n_client, root_data_size, kwargs['alpha'])