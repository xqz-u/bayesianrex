"""Simple utility to load demonstrations in Pytorch style for training."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

StatesActionsLabels = Dict[str, Union[np.ndarray, List[np.ndarray]]]


# TODO somehow pull arrays from disk when requested, maybe with a generator?
# the whole joblib.load ordeal is painfully slow
class RLDemonstrationsDataset(Dataset):
    data: StatesActionsLabels
    feature_shape: int

    def __init__(
        self,
        demonstrations: Optional[StatesActionsLabels] = None,
        demonstrations_path: Optional[Path] = None,
    ):
        if demonstrations is None:
            assert (
                demonstrations_path is not None
            ), "Either 'demonstrations' or 'demonstrations_path' must be passed"
            demonstrations = joblib.load(demonstrations_path)
        assert set(demonstrations.keys()) == {
            "states",
            "actions",
            "labels",
        }, f"Some demonstration keys missing, got: '{demonstrations.keys()}'"
        self.data = demonstrations
        # features dimensions: state dimensions + action dimensions (1)
        self.feature_shape = demonstrations["states"][0][0].shape[1] + 1

    def __len__(self) -> int:
        return len(self.data["states"])

    # NOTE using both state and action as feature; might also use s,a,s'
    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        s_i, s_j = self.data["states"][idx]
        a_i, a_j = self.data["actions"][idx]
        sa_i = torch.hstack((torch.from_numpy(s_i), torch.from_numpy(a_i[..., None])))
        sa_j = torch.hstack((torch.from_numpy(s_j), torch.from_numpy(a_j[..., None])))
        return (sa_i, sa_j), torch.from_numpy(self.data["labels"][idx][..., None])
