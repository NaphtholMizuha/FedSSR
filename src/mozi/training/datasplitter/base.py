import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

class DataSplitter(ABC):
    def __init__(self, dataset: Dataset, n_client: int, root_data_size: int = 0) -> None:
        """
        Initializes the data splitter.
        """
        self.original_dataset = dataset
        self.n_client = n_client
        self.root_data_size = root_data_size

        # These will be populated by the split method
        self.root_indices = []
        self.client_pool_indices = list(range(len(dataset)))
        self.dataset_for_splitting = self.original_dataset # Dataset passed to calc_split_map

    def _get_targets(self, dataset):
        """Helper to get targets from a dataset, handling Subset objects."""
        if isinstance(dataset, Subset):
            # If it's a subset, get targets from the original dataset using the subset's indices
            return np.array(self.original_dataset.targets)[dataset.indices]
        try:
            return np.array(dataset.targets)
        except AttributeError:
            print("Warning: Dataset has no .targets attribute, iterating to get labels...")
            return np.array([label for _, label in dataset])

    def _create_root_set(self):
        """Performs stratified sampling to create root_indices and client_pool_indices."""
        if self.root_data_size <= 0: return

        print(f"Creating a class-balanced root_set of size {self.root_data_size}...")
        
        all_targets = self._get_targets(self.original_dataset)
        
        class_indices = defaultdict(list)
        for i, label in enumerate(all_targets):
            class_indices[label].append(i)

        num_classes = len(class_indices)
        if num_classes == 0: raise ValueError("Could not find any classes.")
        
        samples_per_class = self.root_data_size // num_classes
        if samples_per_class == 0: raise ValueError(f"root_data_size is too small for {num_classes} classes.")
        
        root_indices = []
        client_pool_indices_set = set(range(len(self.original_dataset)))

        for label, indices in class_indices.items():
            if len(indices) < samples_per_class: raise ValueError(f"Class {label} has insufficient samples.")
            
            chosen_for_root = np.random.choice(indices, samples_per_class, replace=False)
            root_indices.extend(chosen_for_root)
        
        # Handle remainder
        num_missing = self.root_data_size - len(root_indices)
        if num_missing > 0:
            potential_additions = list(client_pool_indices_set - set(root_indices))
            additional_indices = np.random.choice(potential_additions, num_missing, replace=False)
            root_indices.extend(additional_indices)

        self.root_indices = root_indices
        self.client_pool_indices = list(client_pool_indices_set - set(self.root_indices))
        
        print(f"Root set created with {len(self.root_indices)} samples.")
        print(f"Client data pool size: {len(self.client_pool_indices)} samples.")

    @abstractmethod
    def calc_split_map(self):
        """
        Subclasses implement this to split self.dataset_for_splitting.
        Must return a list of lists of indices *relative* to self.dataset_for_splitting.
        """
        pass
    
    def _create_subsets_from_absolute_indices(self, idcs_li):
        subsets = [Subset(self.original_dataset, idcs) for idcs in idcs_li]
        print("Created subsets with lengths:")
        print([(i, len(subsets[i])) for i in range(len(subsets))])
        return subsets
    
    def split(self):
        """
        Main splitting method.
        Returns:
            - (root_subset, client_subsets) if root_data_size > 0
            - client_subsets if root_data_size == 0
        """
        # 1. Create root set if requested, which populates self.root_indices and self.client_pool_indices
        self._create_root_set()

        # 2. Prepare the dataset for the client splitting logic
        self.dataset_for_splitting = Subset(self.original_dataset, self.client_pool_indices)
        
        # 3. Call the subclass's splitting logic (e.g., IID, Dirichlet)
        # This returns indices relative to the client pool (self.dataset_for_splitting)
        client_idcs_map_relative = self.calc_split_map()
        
        # 4. Convert relative client indices to absolute indices
        client_idcs_map_absolute = []
        for relative_indices in client_idcs_map_relative:
            absolute_indices = [self.client_pool_indices[i] for i in relative_indices]
            client_idcs_map_absolute.append(absolute_indices)

        # 5. Create the final Subset objects
        client_subsets = self._create_subsets_from_absolute_indices(client_idcs_map_absolute)

        root_subset = Subset(self.original_dataset, self.root_indices)
        return root_subset, client_subsets
