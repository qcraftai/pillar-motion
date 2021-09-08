import torch.nn.functional as F
import torch.nn as nn
import torch
from ..registry import MOTION
from .. import builder


@MOTION.register_module
class MotionNet(nn.Module):
    def __init__(self, 
                reader,
                backbone,
                neck,
                head,
                voxel_cfg):
        super(MotionNet, self).__init__()

        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.motionhead = builder.build_head(head)
        self.voxel_cfg = voxel_cfg

    def extract_feat(self, data):
        input_features = self.reader(
                data["features"], data["num_voxels"], data["coors"]
            )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        x = self.neck(x)
        return x

    def forward(self, example):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )


        # Motion Displacement prediction
        x = self.extract_feat(data)
        preds = self.motionhead(x)

        if self.training:
            return self.motionhead.loss(example, preds, self.voxel_cfg)
        else:
            return self.motionhead.pred(example, preds)
