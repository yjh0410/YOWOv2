import torch
import torch.nn as nn

from ..basic.conv import Conv2d


class DecoupledHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        self.num_cls_heads = cfg['num_cls_heads']
        self.num_reg_heads = cfg['num_reg_heads']
        self.act_type = cfg['head_act']
        self.norm_type = cfg['head_norm']
        self.head_dim = cfg['head_dim']
        self.depthwise = cfg['head_depthwise']

        self.cls_head = nn.Sequential(*[
            Conv2d(self.head_dim, 
                   self.head_dim, 
                   k=3, p=1, s=1, 
                   act_type=self.act_type, 
                   norm_type=self.norm_type,
                   depthwise=self.depthwise)
                   for _ in range(self.num_cls_heads)])
        self.reg_head = nn.Sequential(*[
            Conv2d(self.head_dim, 
                   self.head_dim, 
                   k=3, p=1, s=1, 
                   act_type=self.act_type, 
                   norm_type=self.norm_type,
                   depthwise=self.depthwise)
                   for _ in range(self.num_reg_heads)])


    def forward(self, cls_feat, reg_feat):
        cls_feats = self.cls_head(cls_feat)
        reg_feats = self.reg_head(reg_feat)

        return cls_feats, reg_feats


def build_head(cfg):
    return DecoupledHead(cfg)
    