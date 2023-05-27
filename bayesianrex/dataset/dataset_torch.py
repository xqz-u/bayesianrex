"""Simple utility to load demonstrations in Pytorch style for training."""
from typing import List, Tuple, Union

from bayesianrex.utils import tensorify
from bayesianrex.dataset.generate_demonstrations import TrainTrajectories
from torch import Tensor as T
from torch.utils.data import DataLoader, Dataset

DemonstrationDatapoint = Union[Tuple[T, T], T]


class RLDemonstrationsDataset(Dataset):
    datapoints: List[DemonstrationDatapoint]

    def __init__(self, demonstrations: TrainTrajectories):
        self.datapoints = tensorify(list(zip(*demonstrations)))

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> List[DemonstrationDatapoint]:
        return self.datapoints[idx]


def make_demonstrations_loader(data: TrainTrajectories, shuffle=True) -> DataLoader:
    return DataLoader(RLDemonstrationsDataset(data), shuffle=shuffle, batch_size=1)
