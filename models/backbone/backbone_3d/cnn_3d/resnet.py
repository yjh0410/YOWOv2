import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


model_urls = {
    "resnet18": "https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/resnet-18-kinetics.pth",
    "resnet34": "https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/resnet-34-kinetics.pth",
    "resnet50": "https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/resnet-50-kinetics.pth",
    "resnet101": "https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/resnet-101-kinetics.pth"
}



def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    zero_pads = zero_pads.to(out.data.device)
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        # self.avgpool = nn.AvgPool3d((2, 1, 1), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)

        c2 = self.layer1(c2)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        if c5.size(2) > 1:
            c5 = torch.mean(c5, dim=2, keepdim=True)
        
        return c5.squeeze(2)


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
                # print(k)
        else:
            new_state_dict.pop(k)
            # print(k)

    model.load_state_dict(new_state_dict)
        
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a 3D ResNet-18 model."""

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnet18')

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a 3D ResNet-34 model."""

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnet34')

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a 3D ResNet-50 model. """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnet50')

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a 3D ResNet-101 model."""

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnet101')

    return model


# build 3D resnet
def build_resnet_3d(model_name='resnet18', pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained, shortcut_type='A')
        feats = 512

    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained, shortcut_type='B')
        feats = 2048

    elif model_name == 'resnet101':
        model = resnet101(pretrained=pretrained, shortcut_type='b')
        feats = 2048

    return model, feats


if __name__ == '__main__':
    import time
    model, feats = build_resnet_3d(model_name='resnet18', pretrained=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    x = torch.randn(1, 3, 16, 64, 64).to(device)
    # star time
    t0 = time.time()
    out = model(x)
    print('time', time.time() - t0)

    print(out.shape)
