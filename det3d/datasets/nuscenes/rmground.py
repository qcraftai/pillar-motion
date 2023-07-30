import os
import numpy as np
from tqdm import tqdm
import fire

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from tools.groundseg import segmentation

def nusc_rmgrnd(nusc_root, save_path, version='v1.0-trainval'):
    nusc = NuScenes(version, nusc_root)
    sds = [sd for sd in nusc.sample_data if sd["channel"] == "LIDAR_TOP"]
    for sd in tqdm(sds):
        pc = LidarPointCloud.from_file(f"{nusc_root}/{sd['filename']}")
        pts = np.array(pc.points[:4].T)
        lbls = np.full(len(pts), 30, dtype=np.uint8)
        ego_mask = np.logical_and(
                np.logical_and(-0.8 <= pc.points[0], pc.points[0] <= 0.8),
                np.logical_and(-1.5 <= pc.points[1], pc.points[1] <= 2.5)
            )
        lbls[ego_mask] = 31
        index = np.flatnonzero(np.logical_not(ego_mask))
        label = segmentation.segment(pts[index, :3])

        grnd_index = np.flatnonzero(label)
        lbls[index[grnd_index]] = 24
        # pts = pts[lbls==30]
        lbls = (lbls==30).reshape((-1,1))
        pts = np.concatenate((pts, lbls), axis=1)
        pts = pts.astype(np.float32)
        pts = pts.reshape(-1)
        pts.tofile(os.path.join(save_path, sd['filename']))

if __name__ == '__main__':
   fire.Fire()