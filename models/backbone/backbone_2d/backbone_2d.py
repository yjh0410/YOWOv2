import torch.nn as nn
from .cnn_2d import build_2d_cnn


class Backbone2D(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg

        self.backbone, self.feat_dims = build_2d_cnn(cfg, pretrained)

        
    def forward(self, x):
        """
            Input:
                x: (Tensor) -> [B, C, H, W]
            Output:
                y: (List) -> [
                    (Tensor) -> [B, C1, H1, W1],
                    (Tensor) -> [B, C2, H2, W2],
                    (Tensor) -> [B, C3, H3, W3]
                    ]
        """
        feat = self.backbone(x)

        return feat
