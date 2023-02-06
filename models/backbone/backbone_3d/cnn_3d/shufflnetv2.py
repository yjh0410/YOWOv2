'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['resnext50', 'resnext101', 'resnet152']


model_urls = {
    "0.25x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_0.25x_RGB_16_best.pth",
    "1.0x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_1.0x_RGB_16_best.pth",
    "1.5x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_1.5x_RGB_16_best.pth",
    "2.0x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_2.0x_RGB_16_best.pth",
}


# basic component
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x
    

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.stride == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True)
            )
        
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True)
            )
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True)
            )


    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        


    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :, :]
            x2 = x[:, (x.shape[1]//2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


# ShuffleNet-v2
class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult='1.0x', num_classes=600):
        super(ShuffleNetV2, self).__init__()
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == '0.25x':
            self.stage_out_channels = [-1, 24,  32,  64, 128]
        elif width_mult == '0.5x':
            self.stage_out_channels = [-1, 24,  48,  96, 192]
        elif width_mult == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464]
        elif width_mult == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704]
        elif width_mult == '2.0x':
            self.stage_out_channels = [-1, 24, 224, 488, 976]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=(1,2,2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # # building last several layers
        # self.conv_last      = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
        # self.avgpool        = nn.AvgPool3d((2, 1, 1), stride=1)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        # out = self.conv_last(out) 

        if x.size(2) > 1:
            x = torch.mean(x, dim=2, keepdim=True)
        
        return x.squeeze(2)


def load_weight(model, arch):
    print('Loading pretrained weight ...')
    url = model_urls[arch]
    # check
    if url is None:
        print('No pretrained weight for 3D CNN: {}'.format(arch.upper()))
        return model

    print('Loading 3D backbone pretrained weight: {}'.format(arch.upper()))
    # checkpoint state dict
    checkpoint = load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    checkpoint_state_dict = checkpoint.pop('state_dict')

    # model state dict
    model_state_dict = model.state_dict()
    # reformat checkpoint_state_dict:
    new_state_dict = {}
    for k in checkpoint_state_dict.keys():
        v = checkpoint_state_dict[k]
        new_state_dict[k[7:]] = v

    # check
    for k in list(new_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(new_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                new_state_dict.pop(k)
                print(k)
        else:
            new_state_dict.pop(k)
            print(k)

    model.load_state_dict(new_state_dict)
        
    return model


# build 3D shufflenet_v2
def build_shufflenetv2_3d(model_size='0.25x', pretrained=False):
    model = ShuffleNetV2(model_size)
    feats = model.stage_out_channels[-1]

    if pretrained:
        model = load_weight(model, model_size)

    return model, feats


if __name__ == '__main__':
    import time
    model, feat = build_shufflenetv2_3d(model_size='1.0x', pretrained=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # [B, C, T, H, W]
    x = torch.randn(1, 3, 16, 64, 64).to(device)
    # star time
    t0 = time.time()
    out = model(x)
    print('time', time.time() - t0)
    print(out.shape)
