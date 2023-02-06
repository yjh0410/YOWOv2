import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from yolo_free_basic import Conv
except:
    from .yolo_free_basic import Conv


class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's head
    """
    def __init__(self, in_dim, out_dim, fpn_size='large', depthwise=False, act_type='silu', norm_type='BN'):
        super(ELANBlock, self).__init__()
        if fpn_size == 'tiny' or fpn_size =='nano':
            e1, e2 = 0.25, 1.0
            width = 2
            depth = 1
        elif fpn_size == 'large':
            e1, e2 = 0.5, 0.5
            width = 4
            depth = 1
        elif fpn_size == 'huge':
            e1, e2 = 0.5, 0.5
            width = 4
            depth = 2
        inter_dim = int(in_dim * e1)
        inter_dim2 = int(inter_dim * e2) 
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.ModuleList()
        for idx in range(width):
            if idx == 0:
                cvs = [Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cvs = [Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            # deeper
            if depth > 1:
                for _ in range(1, depth):
                    cvs.append(Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
                self.cv3.append(nn.Sequential(*cvs))
            else:
                self.cv3.append(cvs[0])

        self.out = Conv(inter_dim*2+inter_dim2*len(self.cv3), out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        inter_outs = [x1, x2]
        for m in self.cv3:
            y1 = inter_outs[-1]
            y2 = m(y1)
            inter_outs.append(y2)

        # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat(inter_outs, dim=1))

        return out


class DownSample(nn.Module):
    def __init__(self, in_dim, depthwise=False, act_type='silu', norm_type='BN'):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# PaFPN-ELAN
class PaFPNELAN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 1024],
                 out_dim=256,
                 fpn_size='large',
                 depthwise=False,
                 norm_type='BN',
                 act_type='silu'):
        super(PaFPNELAN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims
        if fpn_size == 'tiny':
            width = 0.5
        elif fpn_size == 'nano':
            assert depthwise
            width = 0.5
        elif fpn_size == 'large':
            width = 1.0
        elif fpn_size == 'huge':
            width = 1.25

        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c4, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),
                                     fpn_size=fpn_size,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.cv3 = Conv(int(256 * width), int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv4 = Conv(c3, int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELANBlock(in_dim=int(128 * width) + int(128 * width),
                                     out_dim=int(128 * width),  # 128
                                     fpn_size=fpn_size,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # bottom up
        # P3 -> P4
        if fpn_size == 'large' or fpn_size == 'huge':
            self.mp1 = DownSample(int(128 * width), act_type=act_type,
                                  norm_type=norm_type, depthwise=depthwise)
        elif fpn_size == 'tiny':
            self.mp1 = Conv(int(128 * width), int(256 * width), k=3, p=1, s=2,
                                act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        elif fpn_size == 'nano':
            self.mp1 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),
                Conv(int(128 * width), int(256 * width), k=1, act_type=act_type, norm_type=norm_type)
            )
        self.head_elan_3 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),  # 256
                                     fpn_size=fpn_size,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        if fpn_size == 'large' or fpn_size == 'huge':
            self.mp2 = DownSample(int(256 * width), act_type=act_type,
                                  norm_type=norm_type, depthwise=depthwise)
        elif fpn_size == 'tiny':
            self.mp2 = Conv(int(256 * width), int(512 * width), k=3, p=1, s=2,
                                act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        elif fpn_size == 'nano':
            self.mp2 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),
                Conv(int(256 * width), int(512 * width), k=1, act_type=act_type, norm_type=norm_type)
            )
        self.head_elan_4 = ELANBlock(in_dim=int(512 * width) + c5,
                                     out_dim=int(512 * width),  # 512
                                     fpn_size=fpn_size,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        self.head_conv_1 = Conv(int(128 * width), int(256 * width), k=3, p=1,
                                act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_conv_2 = Conv(int(256 * width), int(512 * width), k=3, p=1,
                                act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_conv_3 = Conv(int(512 * width), int(1024 * width), k=3, p=1,
                                act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        # output proj layers
        if self.out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, self.out_dim, k=1,
                     norm_type=norm_type, act_type=act_type)
                     for in_dim in [int(256 * width), int(512 * width), int(1024 * width)]
                     ])


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)
        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)
        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        c20 = self.head_conv_1(c13)
        c21 = self.head_conv_2(c16)
        c22 = self.head_conv_3(c19)

        out_feats = [c20, c21, c22] # [P3, P4, P5]
        
        # output proj layers
        if self.out_dim is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'pafpn_elan':
        fpn_net = PaFPNELAN(in_dims=in_dims,
                            out_dim=out_dim,
                            fpn_size=cfg['fpn_size'],
                            depthwise=cfg['fpn_depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'])
                                                        

    return fpn_net
