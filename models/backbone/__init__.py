from .backbone_2d.backbone_2d import Backbone2D
from .backbone_3d.backbone_3d import Backbone3D


def build_backbone_2d(cfg, pretrained=False):
    backbone = Backbone2D(cfg, pretrained)
    return backbone, backbone.feat_dims


def build_backbone_3d(cfg, pretrained=False):
    backbone = Backbone3D(cfg, pretrained)
    return backbone, backbone.feat_dim

