import sys
import pickle
import json
import random
import operator
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from nuscenes.nuscenes import NuScenes
from motion.datasets.custom import PointCloudDataset

from motion.datasets.registry import DATASETS


@DATASETS.register_module
class NuScenesDataset(PointCloudDataset):
    NumPointFeatures = 4  # x, y, z, intensity, timestamp, (beam_id)

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        test_mode=False,
        version="v1.0-trainval",
        **kwargs,
    ):
        super(NuScenesDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, 
        )

        self._info_path = info_path

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        self._num_point_features = NuScenesDataset.NumPointFeatures

        self.version = version


    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        if isinstance(_nusc_infos_all, dict):
            self._nusc_infos = []
            for v in _nusc_infos_all.values():
                self._nusc_infos.extend(v)
        else:
            self._nusc_infos = _nusc_infos_all

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._nusc_infos)

    

    def get_sensor_data(self, idx):

        info = self._nusc_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)
        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)
