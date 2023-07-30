import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionNet(nn.Module):
    def __init__(self, 
                reader,
                backbone,
                head,
                voxel_cfg):
        super(MotionNet, self).__init__()

        self.reader = reader
        self.backbone = backbone
        self.head = head
        self.voxel_cfg = voxel_cfg

    def extract_feat(self, data):
        x = self.reader(data)
        if self.backbone is not None:
            x = self.backbone(x)
        return x

    def _forward(self, example):
        points = example['points']
        x = self.extract_feat(points)
        return self.head(x)

    def forward(self, example):
        if  self.training:
            return self.training_step(example)
        else:
            return self.validation_step(example)
    
    def training_step(self, example):
        preds = self._forward(example)
        loss, log_vars = self.head.loss(example, preds, self.voxel_cfg)

        return loss, log_vars

    @torch.no_grad()
    def validation_step(self, example):
        preds = self._forward(example)
        return preds