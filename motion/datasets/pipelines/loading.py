import numpy as np
from pathlib import Path


from ..registry import PIPELINES


def read_file(path, num_point_feature=4):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points

def read_sweep(sweep):
    min_distance = 1.0
    
    points_sweep = read_file(str(sweep["lidar_path"])).T

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points))))[:3, :]
    points_sweep = remove_close(points_sweep, min_distance)
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))  # time id

    return points_sweep.T, curr_times.T

def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="NuScenesDataset",  **kwargs):
        self.type = dataset

        self.nsweeps = kwargs.get('nsweeps', -1)
        self.mode = kwargs['mode']

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":
            
            # current sweep
            lidar_path = info["lidar_path"]
            points = read_file(str(lidar_path))

            if self.mode == 'train':
                res['lidar']['source_points'] = points
            
            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            # read previous sweeps and transform to current coordinate
            nsweeps = self.nsweeps
            assert (nsweeps - 1) <= len(info["sweeps"]), "nsweeps {} should not greater than list length {}.".format(
                nsweeps, len(info["sweeps"]))

            for i in range((nsweeps - 1)):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep)
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
            
            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)  
            res["lidar"]["points"] = np.hstack([points, times])
            
            if self.mode == 'train':
                # future points
                next_points = read_file(str(info['next_lidar_path'])).T  # 4, n
                next_points[:3, :] = info["next2curr"].dot(
                    np.vstack((next_points[:3, :], np.ones(next_points.shape[1]))))[:3, :]
                next_points = next_points.T

                res['lidar']['target_points'] = next_points
                

                # load pre-computed point2image projection and optical flow
                res["lidar"]["cam_id"] = np.fromfile(info['cam_id'], dtype=np.int8).reshape([-1, 1]) # camera id [0,1,2,3,4,5] -1 for out of range
                res['lidar']['cam_coords'] = np.fromfile(info['cam_coords'], dtype=np.int32).reshape([-1, 2]) # camera coordinates
                res["lidar"]["flow"] = np.fromfile(info['flow'], dtype=np.float32).reshape([-1, 3])  # u, v, valid

        
        else:
            raise NotImplementedError

        return res, info
