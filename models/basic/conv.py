import torch.nn as nn


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


# 2D Conv
def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
    return conv


def get_norm2d(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(dim)


class Conv2d(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 g=1,
                 act_type='',          # activation
                 norm_type='',         # normalization
                 depthwise=False):
        super(Conv2d, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            assert c1 == c2, "In depthwise conv, the in_dim (c1) should be equal to out_dim (c2)."
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm2d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm2d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=g, bias=add_bias))
            if norm_type:
                convs.append(get_norm2d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# 3D Conv
def get_conv3d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv3d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
    return conv


def get_norm3d(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm3d(dim)
    elif norm_type == 'IN':
        return nn.InstanceNorm3d(dim)


class Conv3d(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 g=1,
                 act_type='',          # activation
                 norm_type='',         # normalization
                 depthwise=False):
        super(Conv3d, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            assert c1 == c2, "In depthwise conv, the in_dim (c1) should be equal to out_dim (c2)."
            convs.append(get_conv3d(c1, c2, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm3d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv3d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm3d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv3d(c1, c2, k=k, p=p, s=s, d=d, g=g, bias=add_bias))
            if norm_type:
                convs.append(get_norm3d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)

