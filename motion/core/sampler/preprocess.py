import abc
import sys
import time
from collections import OrderedDict
from functools import reduce


import numpy as np


import copy


def random_flip_xy(points, trans, probability=0.5):
    # x flip
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        points[:, 1] = -points[:, 1]
        back = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for i in range(6):
            trans[i] = trans[i] @ back
    # y flip
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        points[:, 0] = -points[:, 0]
        back = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for i in range(6):
            trans[i] = trans[i] @ back
    return points, trans


def global_scaling(points, trans, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    back = np.array([[1./noise_scale, 0, 0, 0], [0, 1./noise_scale, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    for i in range(6):
        trans[i] = trans[i] @ back
    return points, trans
