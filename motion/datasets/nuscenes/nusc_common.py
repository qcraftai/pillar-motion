import os.path as osp
import os
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List

from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval

from motion.datasets.nuscenes.flow import save_flow

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT','CAM_BACK_RIGHT']
IMG_WIDTH = 1600
IMG_HEIGHT = 900


def box_velocity(
    nusc, sample_annotation_token: str, max_time_diff: float = 1.5
) -> np.ndarray:
    """
    Estimate the velocity for an annotation.
    If possible, we compute the centered difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the velocity cannot be estimated, values are set to np.nan.
    :param sample_annotation_token: Unique sample_annotation identifier.
    :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
    :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
    """

    current = nusc.get("sample_annotation", sample_annotation_token)

    has_next = current["next"] != ""

    if has_next:
        first = current
        last = nusc.get("sample_annotation", current["next"])
    else:
        return np.array([np.nan, np.nan, np.nan])


    pos_last = np.array(last["translation"])
    pos_first = np.array(first["translation"])
    pos_diff = pos_last - pos_first

   
    return pos_diff 


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



def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        box_list.append(box)
    return box_list




def get_sample_ground_plane(root_path, version):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    rets = {}

    for sample in tqdm(nusc.sample):
        chan = "LIDAR_TOP"
        sd_token = sample["data"][chan]
        sd_rec = nusc.get("sample_data", sd_token)

        lidar_path = nusc.get_sample_data_path(nusc, sd_token)
        points = read_file(lidar_path)
        points = np.concatenate((points[:, :3], np.ones((points.shape[0], 1))), axis=1)

        plane, inliers, outliers = fit_plane_LSE_RANSAC(
            points, return_outlier_list=True
        )

        xx = points[:, 0]
        yy = points[:, 1]
        zz = (-plane[0] * xx - plane[1] * yy - plane[3]) / plane[2]

        rets.update({sd_token: {"plane": plane, "height": zz,}})

    with open(nusc.root_path / "infos_trainval_ground_plane.pkl", "wb") as f:
        pickle.dump(rets, f)

def _fill_trainval_infos(nusc,
                         root_path,
                         nsweeps=5,
                         step=4,
                         train_nsweeps_back=30,
                         val_nsweeps_back=25,
                         nsweeps_future=20):
    train_nusc_infos = []
    val_nusc_infos = []
    test_nusc_infos = []

    scenes = pickle.load(open('motion/datasets/nuscenes/split.p','rb'))
    train_scenes = scenes.get('train')
    val_scenes = scenes.get('val')
    test_scenes = scenes.get('test')
    
    sample_info_dict = get_sample_infos(nusc, nsweeps, step)

    for sample in tqdm(nusc.sample):
        """ Manual save info["sweeps"] """
        sample_data_token = sample["data"]["LIDAR_TOP"]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        if sample["scene_token"] in train_scenes:
            least_back = train_nsweeps_back
        else:
            least_back = val_nsweeps_back
        
        flag = True
        for _ in range(1, least_back):
            if curr_sd_rec["prev"] == '':
                flag = False
                break
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])
        
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        for _ in range(nsweeps_future):
            if curr_sd_rec["next"] == '':
                flag = False
                break
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["next"])

        if not flag:
            continue

        info = sample_info_dict[sample['token']]
        
        ## split the data
        if sample["scene_token"] in train_scenes:
            ## saving for next frame
            sample_next = sample_info_dict[sample['next']]
            info['next_lidar_path'] = sample_next['lidar_path']
            info['next_timestamp'] = sample_next['timestamp']
            # transform lidar in the next frame to current frame coordinates
            info['next2curr'] = reduce(np.dot, (info['ref_from_car'],
                                                info["car_from_global"],
                                                np.linalg.inv(sample_next["car_from_global"]),
                                                np.linalg.inv(sample_next["ref_from_car"])))
            info['next_cam_intrinsic_list'] = sample_next['cam_intrinsic_list']
            
            # lidar coordinate (t) ----> camera coordinates (t+1)
            lidar_to_next_cam = []
            for cid in range(len(CAM_NAMES)):
                lidar_to_next_cam.append(reduce(np.dot, (
                            sample_next['cam_from_car_list'][cid],
                            sample_next['cam_car_from_global_list'][cid],
                            np.linalg.inv(info['car_from_global']),
                            np.linalg.inv(info['ref_from_car']))))
            info['lidar_to_next_cam'] = lidar_to_next_cam
            info['next_cam_path_list'] = sample_next["cam_path_list"]
            ## camera projection
            info = lidar2cam(info, root_path)

            ## optical flow preparation
            info = save_flow(info, root_path)

            train_nusc_infos.append(info)
            
        elif sample["scene_token"] in val_scenes:
            val_nusc_infos.append(info)
        else:
            test_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos, test_nusc_infos




def get_sample_infos(nusc, nsweeps=5, step=4):
    """
    get sample info for each key frame
    """
    from nuscenes.utils.geometry_utils import transform_matrix

    sample_info_dict = {}
    
    for sample in tqdm(nusc.sample):
        '''lidar key frame'''
        lidar_sd_token = sample["data"]["LIDAR_TOP"]
        lidar_sd_rec = nusc.get("sample_data", lidar_sd_token)
        lidar_cs_rec = nusc.get(
            "calibrated_sensor", lidar_sd_rec["calibrated_sensor_token"]
        )
        lidar_pose_rec = nusc.get("ego_pose", lidar_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * lidar_sd_rec["timestamp"]

        ref_lidar_path = nusc.get_sample_data_path(lidar_sd_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            lidar_cs_rec["translation"], Quaternion(lidar_cs_rec["rotation"]),
            inverse=True,
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            lidar_pose_rec["translation"],
            Quaternion(lidar_pose_rec["rotation"]),
            inverse=True,
        )

        # add camera info
        cam_path_list, cam_intrinsic_list, cam_car_from_global_list, cam_from_car_list = [], [], [], []
        for cam_name in CAM_NAMES:
            cam_token = sample['data'][cam_name]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam = nusc.get('sample_data', cam_token)
            cam_pose = nusc.get('ego_pose', cam['ego_pose_token'])
            cam_car_from_global = transform_matrix(
                cam_pose['translation'],
                Quaternion(cam_pose['rotation']),
                inverse=True,
                )
            cam_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            cam_from_car = transform_matrix(
                cam_record["translation"], Quaternion(cam_record["rotation"]),
                inverse=True,
            )

            cam_path_list.append(cam_path)
            cam_intrinsic_list.append(cam_intrinsic)
            cam_car_from_global_list.append(cam_car_from_global)
            cam_from_car_list.append(cam_from_car)

        info = {
            "lidar_path": ref_lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
            ## cameras
            "cam_path_list": cam_path_list,
            "cam_intrinsic_list": cam_intrinsic_list,
            "cam_car_from_global_list": cam_car_from_global_list,
            "cam_from_car_list": cam_from_car_list,
        }

        # sweep info
        sample_data_token = sample["data"]["LIDAR_TOP"]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []
        prev_id = 0
        while len(sweeps) < nsweeps - 1:
            if curr_sd_rec["prev"] == "":
                break
            else:
                # move to previous sweep
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])
                prev_id += 1
                if prev_id % step: # only keep for key frames
                    continue
                # Get past pose
                current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
                global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
                )
                car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False,
                )

                tm = reduce(
                    np.dot,
                    [ref_from_car, car_from_global, global_from_car, car_from_current],
                )

                lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        '''
        # using gt boxes for gt motion field
        annotations = [ nusc.get("sample_annotation", token) for token in sample["anns"]]
        '''
        info["scene_token"] = sample["scene_token"]

        sample_info_dict[sample['token']] = info

    return sample_info_dict


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def create_nuscenes_infos(root_path, nsweeps=5):
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_path, verbose=True)

    train_nusc_infos, val_nusc_infos, test_nusc_infos = _fill_trainval_infos(nusc, root_path, nsweeps=nsweeps)

    print(f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)},  test sample: {len(test_nusc_infos)}")
    
    root_path = Path(root_path)
    with open(root_path / "infos_train_{:02d}sweeps_twoframe_withcam.pkl".format(nsweeps), "wb") as f:
        pickle.dump(train_nusc_infos, f)
    with open(root_path / "infos_val_{:02d}sweeps_twoframe_withcam.pkl".format(nsweeps), "wb") as f:
        pickle.dump(val_nusc_infos, f)
    with open(root_path / "infos_test_{:02d}sweeps_twoframe_withcam.pkl".format(nsweeps), "wb") as f:
        pickle.dump(test_nusc_infos, f)



def lidar2cam(info, root_path):
    lidar_path = info["lidar_path"]
    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)[:, :3]

    map_to_image = np.zeros((points.shape[0], 3), dtype=np.float32) - 1
    points= points.T
    
    for cam_id, _ in enumerate(CAM_NAMES):
        p2cam = reduce(np.dot, [
                    info['cam_from_car_list'][cam_id], info['cam_car_from_global_list'][cam_id],
                    np.linalg.inv(info['car_from_global']),
                    np.linalg.inv(info['ref_from_car'])  # transform from lidar coordinate to image coordinate
                ])
        p_cf = (p2cam @ np.vstack((points, np.ones(points.shape[1]))))[:3, :]
        p_cf = info["cam_intrinsic_list"][cam_id] @ p_cf
        p_cf[:2] = p_cf[:2] / (p_cf[2:3, :] + 1e-8)

        # points projected to current camera
        in_cam = reduce(np.logical_and, (p_cf[0, :] > 1, p_cf[1, :] > 1, 
                                            p_cf[0, :] < IMG_WIDTH - 1, p_cf[1,:] < IMG_HEIGHT - 1, p_cf[2,:] >= 1))
        
        map_to_image[in_cam, 0] = cam_id
        map_to_image[in_cam, 1] = np.around(p_cf[0, in_cam])
        map_to_image[in_cam, 2] = np.around(p_cf[1, in_cam])

    new_path = info['lidar_path'].split('/')[-1]
    cam_id = map_to_image[:, 0].astype(np.int8)
    cam_coord = map_to_image[:, 1:]
    if not osp.exists(osp.join(root_path, 'cam_id')):
        os.mkdir(osp.join(root_path, 'cam_id'))
    if not osp.exists(osp.join(root_path, 'cam_coords')):
        os.mkdir(osp.join(root_path, 'cam_coords'))
    info['cam_id'] = osp.join(root_path, 'cam_id', new_path)
    info['cam_coords'] = osp.join(root_path, 'cam_coords', new_path)
    cam_id.tofile(osp.join(root_path, 'cam_id', new_path))
    cam_coord = cam_coord.reshape(-1)
    cam_coord.tofile(osp.join(root_path, 'cam_coords', new_path))

    return info
