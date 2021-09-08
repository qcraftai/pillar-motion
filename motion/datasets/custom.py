import os.path as osp
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .registry import DATASETS
from .pipelines import Compose

@DATASETS.register_module
class PointCloudDataset(Dataset):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    NumPointFeatures = -1

    def __init__(
        self,
        root_path,
        info_path,
        pipeline=None,
        test_mode=False,
        **kwrags
    ):
        self._info_path = info_path
        self._root_path = Path(root_path)

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_sensor_data(self, query):
        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        raise NotImplementedError

    def pre_pipeline(self, results):
        raise NotImplementedError

    def prepare_train_input(self, idx):
        raise NotImplementedError


    def prepare_test_input(self, idx):
        raise NotImplementedError
