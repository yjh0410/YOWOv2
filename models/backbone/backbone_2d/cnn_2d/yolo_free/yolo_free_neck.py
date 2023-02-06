import torch
import torch.nn as nn

try:
    from yolo_free_basic import Conv
except:
    from .yolo_free_basic import Conv


# Spatial Pyramid Pooling
class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=[5, 9, 13], norm_type='BN', act_type='relu'):
        super(SPP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                for k in pooling_size
            ]
        )
        
        self.cv2 = Conv(inter_dim*(len(pooling_size) + 1), out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.cv2(x)

        return x


# SPP block with CSP module
class SPPBlock(nn.Module):
    """
        Spatial Pyramid Pooling Block
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 pooling_size=[5, 9, 13],
                 act_type='lrelu',
                 norm_type='BN',
                 depthwise=False
                 ):
        super(SPPBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            SPP(inter_dim, 
                inter_dim, 
                expand_ratio=1.0, 
                pooling_size=pooling_size, 
                act_type=act_type, 
                norm_type=norm_type),
        )
        self.cv3 = Conv(inter_dim * 2, out_dim, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        y = self.cv3(torch.cat([x1, x2], dim=1))

        return y


# SPP block with CSP module
class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 pooling_size=[5, 9, 13],
                 act_type='lrelu',
                 norm_type='BN',
                 depthwise=False
                 ):
        super(SPPBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, p=1, 
                 act_type=act_type, norm_type=norm_type, 
                 depthwise=depthwise),
            SPP(inter_dim, 
                inter_dim, 
                expand_ratio=1.0, 
                pooling_size=pooling_size, 
                act_type=act_type, 
                norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, 
                 act_type=act_type, norm_type=norm_type, 
                 depthwise=depthwise)
        )
        self.cv3 = Conv(inter_dim * 2, out_dim, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        inter_dim = in_dim // 2  # hidden channels
        self.cv1 = Conv(in_dim, inter_dim, k=1)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    # build neck
    if model == 'spp_block':
        neck = SPPBlock(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )
            
    elif model == 'spp_block_csp':
        neck = SPPBlockCSP(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    elif model == 'sppf':
        neck = SPPF(in_dim, out_dim, k=cfg['pooling_size'])


    return neck


if __name__ == '__main__':
    pass
