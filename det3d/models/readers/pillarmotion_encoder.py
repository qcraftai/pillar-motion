"""
PointPillars with hard/dynamic voxelization
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import torch_scatter
from functools import reduce
from .pillar_encoder import PFNLayer


class PillarMotionNet(nn.Module):
    """
    PillarNet.
    The network performs dynamic pillar scatter that convert point cloud into pillar representation
    and extract pillar features

    Reference:
    PointPillars: Fast Encoders for Object Detection from Point Clouds (https://arxiv.org/abs/1812.05784)
    End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds (https://arxiv.org/abs/1910.06528)

    Args:
        num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
    """

    def __init__(self,
                 num_input_features,
                 voxel_size,
                 pc_range,):
        super().__init__()
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

    def forward(self, points):
        """
        Args:
            points: torch.Tensor of size (N, d), format: batch_id, x, y, z, feat1, ...
        """
        device = points.device
        dtype = points.dtype

        # discard out of range points
        grid_size = (self.pc_range[3:] - self.pc_range[:3]
                     )/self.voxel_size  # x,  y, z
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)

        voxel_size = torch.from_numpy(
            self.voxel_size).type_as(points).to(device)
        pc_range = torch.from_numpy(self.pc_range).type_as(points).to(device)

        points_coords = (
            points[:, 1:4] - pc_range[:3].view(-1, 3)) / voxel_size.view(-1, 3)   # x, y, z

        mask = reduce(torch.logical_and, (points_coords[:, 0] >= 0,
                                          points_coords[:, 0] < grid_size[0],
                                          points_coords[:, 1] >= 0,
                                          points_coords[:, 1] < grid_size[1]))

        points = points[mask]
        time_index = points[:, -1:].long()
        points_coords = points_coords[mask]

        points_coords = points_coords.long()
        batch_idx = points[:, 0:1].long()

        points_index = torch.cat((batch_idx, points_coords[:, :2], time_index), dim=1)
        unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)

        points_mean_scatter = torch_scatter.scatter_mean(
            points[:, 1:4], unq_inv, dim=0)

        f_cluster = points[:, 1:4] - points_mean_scatter[unq_inv]

        # Find distance of x, y, and z from pillar center
        f_center = points[:, 1:3] - (points_coords[:, :2].to(dtype) * voxel_size[:2].unsqueeze(0) +
                                     voxel_size[:2].unsqueeze(0) / 2 + pc_range[:2].unsqueeze(0))

        # Combine together feature decorations
        features = torch.cat([points[:, 1:5], f_cluster, f_center], dim=-1)
        # batch_id, t, y, x
        return features, unq[:, [0, 3, 2, 1]], unq_inv, grid_size


class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features,
        num_filters,
        voxel_size,
        pc_range,
        norm_cfg=None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        assert len(num_filters) > 0
        num_input_features += 5

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.feature_output_dim = num_filters[-1]

        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

        self.voxelization = PillarMotionNet(num_input_features, voxel_size, pc_range)

    def forward(self, points):
        features, coords, unq_inv, grid_size = self.voxelization(points)
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)  # num_points, dim_feat

        feat_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]

        batch_size = torch.unique(points[:, 0]).size(0)
        nsweeps = torch.unique(points[:, -1]).size(0)

        batch_canvas = torch.zeros(
            batch_size, nsweeps, grid_size[1], grid_size[0], self.feature_output_dim,
            dtype=feat_max.dtype,
            device=feat_max.device,
        )
        
        batch_canvas[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = feat_max
        batch_canvas = batch_canvas.permute(0, 4, 1, 2, 3).contiguous()

        return batch_canvas


