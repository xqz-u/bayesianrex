"""Simple utility to load demonstrations in Pytorch style for training."""
from typing import List, Tuple, Union

from bayesianrex.utils import tensorify
from bayesianrex.dataset.generate_demonstrations import TrainTrajectories
from torch import Tensor as T
from torch.utils.data import DataLoader, Dataset

DemonstrationDatapoint = Union[Tuple[T, T], T]


class RLDemonstrationsDataset(Dataset):
    """
    Class to store and handle demonstration data
    """
    datapoints: List[DemonstrationDatapoint]

    def __init__(self, demonstrations: TrainTrajectories):
        """
        Initialize the RLDemonstrationsDataset.
        :param demonstrations: Training trajectories consisting of states, actions, and times
        """
        self.datapoints = tensorify(list(zip(*demonstrations)))

    def __len__(self) -> int:
        """ Return length of dataset """
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> List[DemonstrationDatapoint]:
        """ Return item at index idx at dataset """
        return self.datapoints[idx]


def make_demonstrations_loader(data: TrainTrajectories, shuffle=True) -> DataLoader:
    """ Create a DataLoader for RL demonstrations """
    return DataLoader(RLDemonstrationsDataset(data), shuffle=shuffle, batch_size=1)
