import torch.nn as nn
import torch.nn.functional as F

from .cnn_3d import build_3d_cnn


class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, p=1, s=1, depthwise=False):
        super().__init__()
        if depthwise:
            self.convs = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=k, padding=p, stride=s, groups=in_dim, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, out_dim, kernel_size=1, groups=in_dim, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=k, padding=p, stride=s, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.convs(x)
    

class ConvBlocks(nn.Module):
    def __init__(self, in_dim, out_dim, nblocks=1, depthwise=False):
        super().__init__()
        assert in_dim == out_dim

        conv_block = []
        for _ in range(nblocks):
            conv_block.append(
                Conv(in_dim, out_dim, k=3, p=1, s=1, depthwise=depthwise)
            )
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)
    

class Backbone3D(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg

        # 3D CNN
        self.backbone, self.feat_dim = build_3d_cnn(cfg, pretrained)
        
       
    def forward(self, x):
        """
            Input:
                x: (Tensor) -> [B, C, T, H, W]
            Output:
                y: (List) -> [
                    (Tensor) -> [B, C1, H1, W1],
                    (Tensor) -> [B, C2, H2, W2],
                    (Tensor) -> [B, C3, H3, W3]
                    ]
        """
        feat = self.backbone(x)

        return feat
