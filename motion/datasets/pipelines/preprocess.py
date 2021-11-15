import numpy as np

from motion.core.sampler import preprocess as prep
from motion.core.input.voxel_generator import VoxelGenerator

from functools import reduce
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg, pc_range, voxel_size, **kwargs):
        self.shuffle_points = cfg.shuffle_points   

        self.mode = cfg.mode

        if self.mode == "train":
            self.global_scaling_noise = cfg.global_scale_noise

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.num_cam = kwargs.get('num_cam', 6)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if self.mode == 'train':
            points = res["lidar"]["points"]

            if self.shuffle_points:
                np.random.shuffle(points)

            res['lidar']['lidar_to_next_cam'] = np.stack(info['lidar_to_next_cam'])
            res['lidar']['next_cam_intri'] = np.stack(info['next_cam_intrinsic_list'])
       
            [points, res['lidar']['target_points']], res['lidar']['lidar_to_next_cam'] = \
                prep.random_flip_xy([points, res['lidar']['target_points']], res['lidar']['lidar_to_next_cam'])

            [points, res['lidar']['target_points']], res['lidar']['lidar_to_next_cam']  = prep.global_scaling_v2(
                [points, res['lidar']['target_points']], res['lidar']['lidar_to_next_cam'],  *self.global_scaling_noise)
        
            res["lidar"]["points"] = points

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = cfg.max_voxel_num
        self.nsweeps=cfg.nsweeps

        self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.range,
                nsweeps=self.nsweeps, 
                max_num_points=self.max_points_in_voxel,
                max_voxels=self.max_voxel_num,
            )
    def __call__(self, res, info, pred_motion=None):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = np.hstack((self.voxel_generator.grid_size, np.array([self.nsweeps])))

        voxels, coordinates, num_points = self.voxel_generator.generate(res["lidar"]["points"])
     
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)


        res["voxel"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
        )

        return res, info
