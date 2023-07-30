import collections
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .data_container import DataContainer


def collate(batch_list, samples_per_gpu=1):
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))    
        elif key == "metadata":
            ret[key] = elems
        
        elif key in ["coordinates", "points", "cam_id", "flow", "cam_coords", "source_points", "target_points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))
        elif key in ["motion", "lidar_to_next_cam", 'next_cam_intri']:
            ret[key] = torch.tensor(np.stack(elems, axis=0)).float()
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret