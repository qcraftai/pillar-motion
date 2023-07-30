import torch
import torch.nn as nn


class SingleStageDetector(nn.Module):
    def __init__(self,
                 reader,
                 backbone=None,
                 neck=None,
                 head=None,
                 post_processing=None,
                 **kwargs):

        super(SingleStageDetector, self).__init__()
        self.reader = reader
        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.post_processing = post_processing

    def extract_feat(self, data):
        x = self.reader(data)
        if self.backbone is not None:
            x = self.backbone(*x)
        if self.neck is not None:
            x = self.neck(x)
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
        loss, log_vars = self.head.loss(example, preds)

        return loss, log_vars

    @torch.no_grad()
    def validation_step(self, example):
        preds = self._forward(example)
        outputs = self.head.predict(example, preds, self.post_processing)
        detections = {}
        for output in outputs:
            token = output["token"]
            for k, v in output.items():
                if k != "token":
                    output[k] = v.to(torch.device("cpu"))

            detections.update({token: output})
        return detections
