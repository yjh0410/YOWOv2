import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

try:
    from .yolo_free_backbone import build_backbone
    from .yolo_free_neck import build_neck
    from .yolo_free_fpn import build_fpn
    from .yolo_free_head import build_head
except:
    from yolo_free_backbone import build_backbone
    from yolo_free_neck import build_neck
    from yolo_free_fpn import build_fpn
    from yolo_free_head import build_head


__all__ = ['build_yolo_free']


model_urls = {
    'yolo_free_nano': 'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_coco.pth',
    'yolo_free_tiny': 'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_coco.pth',
    'yolo_free_large': 'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_large_coco.pth',
}


yolo_free_config = {
    'yolo_free_nano': {
        # model
        'backbone': 'shufflenetv2_1.0x',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'anchor_size': None,
        # neck
        'neck': 'sppf',
        'neck_dim': 232,
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'nano',
        'fpn_dim': [116, 232, 232],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': True,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        },

    'yolo_free_tiny': {
        # model
        'backbone': 'elannet_tiny',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 256,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'tiny', # 'tiny', 'large', 'huge
        'fpn_dim': [128, 256, 256],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        },

    'yolo_free_large': {
        # model
        'backbone': 'elannet_large',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 512,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'large', # 'tiny', 'large', 'huge
        'fpn_dim': [512, 1024, 512],
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        },

}


# Anchor-free YOLO
class FreeYOLO(nn.Module):
    def __init__(self, cfg):
        super(FreeYOLO, self).__init__()
        # --------- Basic Config -----------
        self.cfg = cfg

        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dim = build_backbone(self.cfg['backbone'])

        ## neck
        self.neck = build_neck(cfg=self.cfg, in_dim=bk_dim[-1], out_dim=self.cfg['neck_dim'])
        
        ## fpn
        self.fpn = build_fpn(cfg=self.cfg, in_dims=self.cfg['fpn_dim'], out_dim=self.cfg['head_dim'])

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg) 
            for _ in range(len(cfg['stride']))
            ])

    def forward(self, x):
        # backbone
        feats = self.backbone(x)

        # neck
        feats['layer4'] = self.neck(feats['layer4'])

        # fpn
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_cls_feats = []
        all_reg_feats = []
        for feat, head in zip(pyramid_feats, self.non_shared_heads):
            # [B, C, H, W]
            cls_feat, reg_feat = head(feat)

            all_cls_feats.append(cls_feat)
            all_reg_feats.append(reg_feat)

        return all_cls_feats, all_reg_feats


# build FreeYOLO
def build_yolo_free(model_name='yolo_free_large', pretrained=False):
    # model config
    cfg = yolo_free_config[model_name]

    # FreeYOLO
    model = FreeYOLO(cfg)
    feat_dims = [model.cfg['head_dim']] * 3

    # Load COCO pretrained weight
    if pretrained:
        url = model_urls[model_name]

        # check
        if url is None:
            print('No 2D pretrained weight ...')
            return model, feat_dims
        else:
            print('Loading 2D backbone pretrained weight: {}'.format(model_name.upper()))

            # state dict
            checkpoint = load_state_dict_from_url(url, map_location='cpu')
            checkpoint_state_dict = checkpoint.pop('model')

            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        # print(k)
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    # print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

    return model, feat_dims


if __name__ == '__main__':
    model, fpn_dim = build_yolo_free(model_name='yolo_free_nano', pretrained=True)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    cls_feats, reg_feats = model(x)

    for cls_feat, reg_feat in zip(cls_feats, reg_feats):
        print(cls_feat.shape, reg_feat.shape)
