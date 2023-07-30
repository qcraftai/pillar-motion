import json
import operator
import numpy as np
import os
import itertools
from pathlib import Path

from nuscenes import NuScenes

from det3d.datasets.base import BaseDataset


class NuScenesMotionDataset(BaseDataset):
    def __init__(self,
                 info_path,
                 root_path,
                 loading_pipelines=None,
                 augmentation=None,
                 prepare_label=None,):

        super(NuScenesMotionDataset, self).__init__(
            root_path, info_path, loading_pipelines=loading_pipelines, augmentation=augmentation)

    def read_file(self, path, num_point_feature=4):
        points = np.fromfile(os.path.join(self._root_path, 'grndseg', path),
                             dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        return points

    def read_sweep(self, sweep, time_index, min_distance=1.0):
        points_sweep = self.read_file(str(sweep["lidar_path"])).T

        nbr_points = points_sweep.shape[1]
        if sweep["transform_matrix"] is not None:
            points_sweep[:3, :] = sweep["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(nbr_points))))[:3, :]
        points_sweep = self.remove_close(points_sweep, min_distance)
        #curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
        curr_times = time_index * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, curr_times.T

    @staticmethod
    def remove_close(points, radius: float):
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close]
        return points

    def load_pointcloud(self, res, info):
        points = self.read_file(info["lidar_path"],num_point_feature=5)
        is_ground = points[:, -1]
        
        points = points[:, :4]
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for i in range(len(info["sweeps"])):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = self.read_sweep(sweep, i+1)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        res["points"] = np.hstack([points, times])
        info['ground'] = is_ground.astype(bool)

        return res
    
    def load_target(self, res, info):
        # load pre-computed point2image projection and optical flow
        # camera id [0,1,2,3,4,5] -1 for out of range
        cam_proj =  np.fromfile(os.path.join(self._root_path, 'cam_proj', info['lidar_path']),
                                dtype=np.int32).reshape([-1, 3])
        cam_proj = cam_proj[info['ground']]
        res["points_cam_id"] = cam_proj[:, 0:1]
        res['points_cam_coords'] = cam_proj[:, 1:]
        res['lidar_to_next_cam'] = np.stack(info['lidar_to_next_cam'])
        res['next_cam_intri'] = np.stack(info['next_cam_intrinsic_list'])

        flow = np.fromfile(os.path.join(self._root_path, 'points_flow', info['lidar_path']),
                                  dtype=np.float32).reshape([-1, 2]) 
        res["points_flow"] = flow[info['ground']]

        nonground_points = res['points'][res['points'][:, -1] == 0, :3]
        res['source_points'] = nonground_points[info['ground']]

        next_points = self.read_file(info['next_lidar_path'], 5)  # 4, n
        next_points = next_points[next_points[:, -1]==1, :3]
        next_points = next_points.T
        next_points[:3, :] = info["next2curr"].dot(
            np.vstack((next_points[:3, :], np.ones(next_points.shape[1]))))[:3, :]
        next_points = next_points.T

        res['target_points'] = next_points

        return res  
    