import torch
import torch.nn as nn

try:
    from yolo_free_basic import Conv
except:
    from .yolo_free_basic import Conv


class DecoupledHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        self.num_cls_head=cfg['num_cls_head']
        self.num_reg_head=cfg['num_reg_head']
        self.act_type=cfg['head_act']
        self.norm_type=cfg['head_norm']
        self.head_dim = cfg['head_dim']

        self.cls_feats = nn.Sequential(*[Conv(self.head_dim, 
                                              self.head_dim, 
                                              k=3, p=1, s=1, 
                                              act_type=self.act_type, 
                                              norm_type=self.norm_type,
                                              depthwise=cfg['head_depthwise']) for _ in range(self.num_cls_head)])
        self.reg_feats = nn.Sequential(*[Conv(self.head_dim, 
                                              self.head_dim, 
                                              k=3, p=1, s=1, 
                                              act_type=self.act_type, 
                                              norm_type=self.norm_type,
                                              depthwise=cfg['head_depthwise']) for _ in range(self.num_reg_head)])


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats


# build detection head
def build_head(cfg):
    head = DecoupledHead(cfg) 

    return head
    