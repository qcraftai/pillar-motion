import copy
from pathlib import Path
import pickle

import fire

from motion.datasets.nuscenes.nusc_common import create_nuscenes_infos


def nuscenes_data_prep(root_path,  nsweeps=5):
    create_nuscenes_infos(root_path, nsweeps=nsweeps)
   

if __name__ == "__main__":
    fire.Fire()
