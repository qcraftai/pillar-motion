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

# from motion.datasets.nuscenes.flow import save_flow

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT','CAM_BACK_RIGHT']
IMG_WIDTH = 1600
IMG_HEIGHT = 900


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

    scenes = pickle.load(open('det3d/datasets/nuscenes/split.p','rb'))
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
            #info = lidar2cam(info, root_path)

            ## optical flow preparation
            # info = save_flow(info, root_path)

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
            "lidar_path": lidar_sd_rec['filename'],
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


                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "lidar_path": curr_sd_rec['filename'],
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps
        info["scene_token"] = sample["scene_token"]
        # gt boxes for motion field
        boxes, next_boxes = get_boxes(nusc, sample)
        info['boxes'] = boxes
        info['next_boxes'] = next_boxes

        sample_info_dict[sample['token']] = info

    return sample_info_dict

def get_boxes(nusc, sample):
    sample_data_token = sample["data"]["LIDAR_TOP"]
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor",
                            sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
    boxes = []
    next_boxes = []
    for sample_annotation_token in sample['anns']:
        record = nusc.get('sample_annotation', sample_annotation_token)
        box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                    name=record['category_name'], token=record['token'])
        next_box = None
        if record['next'] != '':
            next_record = nusc.get("sample_annotation", record["next"])
            if next_record['sample_token'] == sample['next']:
                next_box = Box(next_record['translation'], next_record['size'], 
                            Quaternion(next_record['rotation']),
                        name=next_record['category_name'], token=next_record['token'])
        boxes.append(box)
        next_boxes.append(next_box)

    for idx in range(len(boxes)):
        box = boxes[idx]
        next_box = next_boxes[idx]

        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        boxes[idx] = box
        
        if next_box is not None:
            next_box.translate(-np.array(pose_record["translation"]))
            next_box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system
            next_box.translate(-np.array(cs_record["translation"]))
            next_box.rotate(Quaternion(cs_record["rotation"]).inverse)
        
            next_boxes[idx] = next_box

    return boxes, next_boxes

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

    new_path = os.path.join(root_path, 'cam_proj', info['lidar_path'])
    map_to_image = map_to_image.reshape(-1).astype(np.int32)
    map_to_image.tofile(new_path)

    return info

def gt_motion_flow(points, curr_box, next_box, Box):
    curr_box.orientation = curr_box.orientation.normalised
    next_box.orientation = next_box.orientation.normalised

    delta_rotation = curr_box.orientation.inverse * next_box.orientation
    rotated_pc = (delta_rotation.rotation_matrix @ points.T).T
    rotated_curr_center = np.dot(delta_rotation.rotation_matrix, curr_box.center)
    delta_center = next_box.center - rotated_curr_center

    rotated_tranlated_pc = rotated_pc + delta_center

    pc_displace_vectors = rotated_tranlated_pc - points

    return pc_displace_vectors