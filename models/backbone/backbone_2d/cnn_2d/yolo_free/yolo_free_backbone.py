import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['build_backbone']

# ======================  ELAN-Net ==========================
# ELANNet
def get_activation(act_type=None):
    if act_type is None:
        return nn.Identity()
    elif act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(in_dim, norm_type=None):
    if norm_type is None:
        return nn.Identity()
    elif norm_type == 'BN':
        return nn.BatchNorm2d(in_dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(32, in_dim)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(in_dim)


class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='silu',
                 norm_type='BN',       # activation
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            # depthwise conv
            convs.append(nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=p, dilation=d, groups=c1, bias=add_bias))
            convs.append(get_norm(c1, norm_type))
            convs.append(get_activation(act_type))

            # pointwise conv
            convs.append(nn.Conv2d(c1, c2, kernel_size=1, stride=s, padding=0, dilation=d, groups=1, bias=add_bias))
            convs.append(get_norm(c2, norm_type))
            convs.append(get_activation(act_type))

        else:
            convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=1, bias=add_bias))
            convs.append(get_norm(c2, norm_type))
            convs.append(get_activation(act_type))

        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, model_size='large', act_type='silu', depthwise=False):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        if model_size == 'tiny':
            depth = 1
        elif model_size == 'large':
            depth = 2
        elif model_size == 'huge':
            depth = 3
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv3 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, depthwise=depthwise)
            for _ in range(depth)
        ])
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, depthwise=depthwise)
            for _ in range(depth)
        ])

        self.out = Conv(inter_dim*4, out_dim, k=1)



    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out


class DownSample(nn.Module):
    def __init__(self, in_dim, act_type='silu', norm_type='BN'):
        super().__init__()
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# ELANNet-Tiny
class ELANNet_Tiny(nn.Module):
    """
    ELAN-Net of YOLOv7-Tiny.
    """
    def __init__(self, depthwise=False):
        super(ELANNet_Tiny, self).__init__()
        
        # tiny backbone
        self.layer_1 = Conv(3, 32, k=3, p=1, s=2, act_type='lrelu', depthwise=depthwise)       # P1/2

        self.layer_2 = nn.Sequential(   
            Conv(32, 64, k=3, p=1, s=2, act_type='lrelu', depthwise=depthwise),             
            ELANBlock(in_dim=64, out_dim=64, expand_ratio=0.5,
                      model_size='tiny', act_type='lrelu', depthwise=depthwise)                  # P2/4
        )
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=64, out_dim=128, expand_ratio=0.5,
                      model_size='tiny', act_type='lrelu', depthwise=depthwise)                  # P3/8
        )
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5,
                      model_size='tiny', act_type='lrelu', depthwise=depthwise)                  # P4/16
        )
        self.layer_5 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),             
            ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5,
                      model_size='tiny', act_type='lrelu', depthwise=depthwise)                  # P5/32
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = {
            'layer2': c3,
            'layer3': c4,
            'layer4': c5
        }
        return outputs


# ELANNet-Large
class ELANNet_Large(nn.Module):
    """
    ELAN-Net of YOLOv7.
    """
    def __init__(self, depthwise=False):
        super(ELANNet_Large, self).__init__()
        
        # large backbone
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1, act_type='silu', depthwise=depthwise),      
            Conv(32, 64, k=3, p=1, s=2, act_type='silu', depthwise=depthwise),
            Conv(64, 64, k=3, p=1, act_type='silu', depthwise=depthwise)                                                   # P1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(64, 128, k=3, p=1, s=2, act_type='silu', depthwise=depthwise),             
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5,
                      model_size='large',act_type='silu', depthwise=depthwise)                     # P2/4
        )
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=256, act_type='silu'),             
            ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5,
                      model_size='large',act_type='silu', depthwise=depthwise)                     # P3/8
        )
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=512, act_type='silu'),             
            ELANBlock(in_dim=512, out_dim=1024, expand_ratio=0.5,
                      model_size='large',act_type='silu', depthwise=depthwise)                    # P4/16
        )
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=1024, act_type='silu'),             
            ELANBlock(in_dim=1024, out_dim=1024, expand_ratio=0.25,
                      model_size='large',act_type='silu', depthwise=depthwise)                  # P5/32
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = {
            'layer2': c3,
            'layer3': c4,
            'layer4': c5
        }
        return outputs


## build ELAN-Net
def build_elannet(model_name='elannet_large'):
    # model
    if model_name == 'elannet_large':
        backbone = ELANNet_Large()
        feat_dims = [512, 1024, 1024]
    elif model_name == 'elannet_tiny':
        backbone = ELANNet_Tiny()
        feat_dims = [128, 256, 512]

    return backbone, feat_dims


# ====================== ShuffleNet-v2 ==========================
# ShuffleNet-v2
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 model_size='1.0x',
                 out_stages=(2, 3, 4),
                 with_last_conv=False,
                 kernal_size=3):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        if model_size == '0.5x':
            self._stage_out_channels = [24, 48, 96, 192]
        elif model_size == '1.0x':
            self._stage_out_channels = [24, 116, 232, 464]
        elif model_size == '1.5x':
            self._stage_out_channels = [24, 176, 352, 704]
        elif model_size == '2.0x':
            self._stage_out_channels = [24, 244, 488, 976]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, self.stage_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleV2Block(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        
        self._initialize_weights()


    def _initialize_weights(self):
        print('init weights...')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = {}
        for i in range(2, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output['layer{}'.format(i)] = x

        return output


## build ShuffleNet-v2
def build_shufflenetv2(model_size='1.0x'):
    """Constructs a shufflenetv2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = ShuffleNetV2(model_size=model_size)
    feat_dims = backbone._stage_out_channels[1:]

    return backbone, feat_dims


# build backbone
def build_backbone(model_name='elannet_large'):
    if model_name in ['elannet_nano', 'elannet_tiny', 'elannet_large', 'elannet_huge']:
        return build_elannet(model_name)

    elif model_name in ['shufflenetv2_0.5x', 'shufflenetv2_1.0x']:
        return build_shufflenetv2(model_size=model_name[-4:])
        

if __name__ == '__main__':
    import time
    model, feats = build_backbone(model_name='shufflenetv2_1.0x')
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for k in outputs.keys():
        print(outputs[k].shape)
