import torch
import torch.nn as nn
from ..basic.conv import Conv2d


# Channel Self Attetion Module
class CSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


# Spatial Self Attetion Module
class SSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(SSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1).permute(0, 2, 1)   # [B, N, C]
        key = x.view(B, C, -1)                      # [B, C, N]
        value = x.view(B, C, -1).permute(0, 2, 1)   # [B, N, C]

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


# Channel Encoder
class ChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            CSAM(),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x1, x2):
        """
            x: [B, C, H, W]
        """
        x = torch.cat([x1, x2], dim=1)
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


# Spatial Encoder
class SpatialEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SSAM(),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        """
            x: [B, C, H, W]
        """
        x = self.fuse_convs(x)

        return x


def build_channel_encoder(cfg, in_dim, out_dim):
    encoder = ChannelEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm']
        )

    return encoder


def build_spatial_encoder(cfg, in_dim, out_dim):
    encoder = SpatialEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm']
        )

    return encoder
