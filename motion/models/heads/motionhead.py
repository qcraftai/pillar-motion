import logging
from collections import defaultdict
import numpy as np
import torch


from torch import nn
from torch.nn import functional as F
from .. import builder

from ..registry import HEADS
from .chamfer import chamfer_distance
from functools import reduce

@HEADS.register_module
class MotionHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(MotionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, 2, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def loss(self, example, pred_motion, voxel_cfg, eps=1e-8, **kwargs):
        batch_size = pred_motion.size(0)
        source_points = example['source_points']
        target_points = example['target_points']

        cam_id = example['cam_id']
        lidar_to_next_cam = example["lidar_to_next_cam"]
        next_cam_intrinsic = example['next_cam_intri']
        points_cam_coords = example["cam_coords"]
        flow = example['flow']

        chamfer_loss = 0
        consistency_loss = 0 

        smooth_loss = torch.mean(torch.abs(pred_motion[:, 1:] - pred_motion[:, :-1])) + \
                      torch.mean(torch.abs(pred_motion[:, :, 1:] - pred_motion[:, :, :-1]))
        

        select = reduce(torch.logical_and, (torch.abs(source_points[:, 1]) < 32,
                                    torch.abs(source_points[:, 2]) < 32))
        source_points = source_points[select]
        cam_id = cam_id[select]
        points_cam_coords = points_cam_coords[select]
        flow = flow[select]

        coord = torch.floor((source_points[:, 1:3] - voxel_cfg.range[0]) / voxel_cfg.voxel_size[0]).long()
        selected_motion = pred_motion[source_points[:, 0].long(), coord[:, 1], coord[:, 0]]
        pred_points_all = source_points[:, 1:4] + \
                        torch.cat((selected_motion, torch.zeros(selected_motion.size(0),1).to(source_points.device)), dim=1)

        with torch.no_grad():
            points_mask = torch.zeros(1, source_points.size(0)).to(source_points.device)
        
        for batchidx in range(batch_size):
            pred_points = pred_points_all[source_points[:, 0] == batchidx]
            batch_targets = target_points[target_points[:, 0] == batchidx, 1:4]
            for camid in range(6):
                cond = torch.logical_and(source_points[:, 0] == batchidx, cam_id[:, 1] == camid)
                if torch.sum(cond) == 0:
                    continue
                with torch.no_grad():
                    select_cam_coords = points_cam_coords[cond]
                    selected_flow = flow[cond]
                    pred_cam_points = pred_points_all[cond]
                
                    ego_points = source_points[cond, 1:4]
                    ego_points_cam = torch.mm(lidar_to_next_cam[batchidx, camid][:3, :3],
                                                ego_points.permute(1, 0).contiguous()) + \
                                                lidar_to_next_cam[batchidx, camid][:3, 3].view(3, 1)  

                    ego_points_cam = torch.mm(next_cam_intrinsic[batchidx, camid], ego_points_cam)
                    ego_points_cam = ego_points_cam[:2] / (ego_points_cam[2:] + eps)
                    ego_motion = ego_points_cam.permute(1, 0).contiguous() - select_cam_coords[:, 1:3]
                    ego_motion = torch.norm(ego_motion - selected_flow[:, 1:3], dim=-1)
                    
                    points_mask[:, cond] = torch.exp(-0.1 * torch.clamp(ego_motion-5.0, min=0.0)) 

                next_points_cam = torch.mm(lidar_to_next_cam[batchidx, camid][:3, :3], pred_cam_points.permute(1, 0).contiguous()) + \
                               lidar_to_next_cam[batchidx, camid][:3, 3].view(3, 1)  # 3, n

                points_to_next_cam = torch.mm(next_cam_intrinsic[batchidx, camid], next_points_cam)
                points_to_next_cam = points_to_next_cam[:2] / (points_to_next_cam[2:] + eps)
                projected_motion = points_to_next_cam.permute(1, 0).contiguous() - select_cam_coords[:, 1:3]  # n  2
                consistency_loss += torch.sum(torch.abs(projected_motion[:, 0:2] - selected_flow[:, 1:3]))/10

            chamfer_loss += chamfer_distance(pred_points.unsqueeze(0), batch_targets.unsqueeze(0),
                                weights=(1.0 - points_mask[:, source_points[:, 0] == batchidx])) 
       
        consistency_loss = consistency_loss / torch.sum(cam_id[:, 1]>=0)
        chamfer_loss /= batch_size
        loss = chamfer_loss + smooth_loss + consistency_loss 
        return loss, {"loss": loss.detach().cpu().item(),
                        "chamfer_loss": chamfer_loss.detach().cpu().item(),
                        "smooth_loss": smooth_loss.detach().cpu().item(),
                        "consistency_loss": consistency_loss.detach().cpu().item(),
                         }

    def predict(self, example, pred_motion, voxel_cfg=None, **kwargs):
        meta = example['metadata'] 

        ret_list = []
        for i in range(len(meta)):
            ret_list.append(dict(token=meta[i]['token'],
                                 pred_motion=pred_motion[i].detach().cpu().numpy()))

        return ret_list
