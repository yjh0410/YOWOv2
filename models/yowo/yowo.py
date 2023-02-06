import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone_2d
from ..backbone import build_backbone_3d
from .encoder import build_channel_encoder
from .head import build_head

from utils.nms import multiclass_nms


# You Only Watch Once
class YOWO(nn.Module):
    def __init__(self, 
                 cfg,
                 device,
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 topk = 50,
                 trainable = False,
                 multi_hot = False):
        super(YOWO, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.multi_hot = multi_hot

        # ------------------ Network ---------------------
        ## 2D backbone
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            cfg, pretrained=cfg['pretrained_2d'] and trainable)
            
        ## 3D backbone
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            cfg, pretrained=cfg['pretrained_3d'] and trainable)

        ## cls channel encoder
        self.cls_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])
            
        ## reg channel & spatial encoder
        self.reg_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])

        ## head
        self.heads = nn.ModuleList(
            [build_head(cfg) for _ in range(len(cfg['stride']))]
        ) 

        ## pred
        head_dim = cfg['head_dim']
        self.conf_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 1, kernel_size=1)
                for _ in range(len(cfg['stride']))
                ]) 
        self.cls_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
                for _ in range(len(cfg['stride']))
                ]) 
        self.reg_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 4, kernel_size=1) 
                for _ in range(len(cfg['stride']))
                ])                 

        # init yowo
        self.init_yowo()


    def init_yowo(self): 
        # Init yolo
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        for conf_pred in self.conf_preds:
            b = conf_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            conf_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def generate_anchors(self, fmp_size, stride):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= stride
        anchors = anchor_xy.to(self.device)

        return anchors
        

    def decode_boxes(self, anchors, pred_reg, stride):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        # size of bbox
        pred_box_wh = pred_reg[..., 2:].exp() * stride

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def post_process_one_hot(self, conf_preds, cls_preds, reg_preds, anchors):
        """
        Input:
            conf_preds: (Tensor) [H x W, 1]
            cls_preds: (Tensor) [H x W, C]
            reg_preds: (Tensor) [H x W, 4]
        """
        
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for level, (conf_pred_i, cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(conf_preds, cls_preds, reg_preds, anchors)):
            # (H x W x C,)
            scores_i = (torch.sqrt(conf_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, reg_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred_i = reg_pred_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return scores, labels, bboxes
    

    def post_process_multi_hot(self, conf_preds, cls_preds, reg_preds, anchors):
        """
        Input:
            cls_pred: (Tensor) [H x W, C]
            reg_pred: (Tensor) [H x W, 4]
        """        
        all_conf_preds = []
        all_cls_preds = []
        all_box_preds = []
        for level, (conf_pred_i, cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(conf_preds, cls_preds, reg_preds, anchors)):
            # decode box
            box_pred_i = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])
            
            # conf pred 
            conf_pred_i = torch.sigmoid(conf_pred_i.squeeze(-1))   # [M,]

            # cls_pred
            cls_pred_i = torch.sigmoid(cls_pred_i)                 # [M, C]

            # topk
            topk_conf_pred_i, topk_inds = torch.topk(conf_pred_i, self.topk)
            topk_cls_pred_i = cls_pred_i[topk_inds]
            topk_box_pred_i = box_pred_i[topk_inds]

            # threshold
            keep = topk_conf_pred_i.gt(self.conf_thresh)
            topk_conf_pred_i = topk_conf_pred_i[keep]
            topk_cls_pred_i = topk_cls_pred_i[keep]
            topk_box_pred_i = topk_box_pred_i[keep]

            all_conf_preds.append(topk_conf_pred_i)
            all_cls_preds.append(topk_cls_pred_i)
            all_box_preds.append(topk_box_pred_i)

        # concatenate
        conf_preds = torch.cat(all_conf_preds, dim=0)  # [M,]
        cls_preds = torch.cat(all_cls_preds, dim=0)    # [M, C]
        box_preds = torch.cat(all_box_preds, dim=0)    # [M, 4]

        # to cpu
        scores = conf_preds.cpu().numpy()
        labels = cls_preds.cpu().numpy()
        bboxes = box_preds.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)

        # [M, 5 + C]
        out_boxes = np.concatenate([bboxes, scores[..., None], labels], axis=-1)

        return out_boxes
    

    @torch.no_grad()
    def inference(self, video_clips):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        return:
        """
        B, _, _, img_h, img_w = video_clips.shape

        # key frame
        key_frame = video_clips[:, :, -1, :, :]
        # 3D backbone
        feat_3d = self.backbone_3d(video_clips)

        # 2D backbone
        cls_feats, reg_feats = self.backbone_2d(key_frame)

        # non-shared heads
        all_conf_preds = []
        all_cls_preds = []
        all_reg_preds = []
        all_anchors = []
        for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
            # upsample
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))

            # encoder
            cls_feat = self.cls_channel_encoders[level](cls_feat, feat_3d_up)
            reg_feat = self.reg_channel_encoders[level](reg_feat, feat_3d_up)

            # head
            cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)

            # pred
            conf_pred = self.conf_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)
        
            # generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C], M = HW
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

            all_conf_preds.append(conf_pred)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)
        
        # batch process
        if self.multi_hot:
            batch_bboxes = []
            for batch_idx in range(video_clips.size(0)):
                cur_conf_preds = []
                cur_cls_preds = []
                cur_reg_preds = []
                for conf_preds, cls_preds, reg_preds in zip(all_conf_preds, all_cls_preds, all_reg_preds):
                    # [B, M, C] -> [M, C]
                    cur_conf_preds.append(conf_preds[batch_idx])
                    cur_cls_preds.append(cls_preds[batch_idx])
                    cur_reg_preds.append(reg_preds[batch_idx])

                # post-process
                out_boxes = self.post_process_multi_hot(
                    cur_conf_preds, cur_cls_preds, cur_reg_preds, all_anchors)

                # normalize bbox
                out_boxes[..., :4] /= max(img_h, img_w)
                out_boxes[..., :4] = out_boxes[..., :4].clip(0., 1.)

                batch_bboxes.append(out_boxes)

            return batch_bboxes

        else:
            batch_scores = []
            batch_labels = []
            batch_bboxes = []
            for batch_idx in range(conf_pred.size(0)):
                # [B, M, C] -> [M, C]
                cur_conf_preds = []
                cur_cls_preds = []
                cur_reg_preds = []
                for conf_preds, cls_preds, reg_preds in zip(all_conf_preds, all_cls_preds, all_reg_preds):
                    # [B, M, C] -> [M, C]
                    cur_conf_preds.append(conf_preds[batch_idx])
                    cur_cls_preds.append(cls_preds[batch_idx])
                    cur_reg_preds.append(reg_preds[batch_idx])

                # post-process
                scores, labels, bboxes = self.post_process_one_hot(
                    cur_conf_preds, cur_cls_preds, cur_reg_preds, all_anchors)

                # normalize bbox
                bboxes /= max(img_h, img_w)
                bboxes = bboxes.clip(0., 1.)

                batch_scores.append(scores)
                batch_labels.append(labels)
                batch_bboxes.append(bboxes)

            return batch_scores, batch_labels, batch_bboxes


    def forward(self, video_clips):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        return:
            outputs: (Dict) -> {
                'pred_conf': (Tensor) [B, M, 1]
                'pred_cls':  (Tensor) [B, M, C]
                'pred_reg':  (Tensor) [B, M, 4]
                'anchors':   (Tensor) [M, 2]
                'stride':    (Int)
            }
        """                        
        if not self.trainable:
            return self.inference(video_clips)
        else:
            # key frame
            key_frame = video_clips[:, :, -1, :, :]
            # 3D backbone
            feat_3d = self.backbone_3d(video_clips)

            # 2D backbone
            cls_feats, reg_feats = self.backbone_2d(key_frame)

            # non-shared heads
            all_conf_preds = []
            all_cls_preds = []
            all_box_preds = []
            all_anchors = []
            for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
                # upsample
                feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))

                # encoder
                cls_feat = self.cls_channel_encoders[level](cls_feat, feat_3d_up)
                reg_feat = self.reg_channel_encoders[level](reg_feat, feat_3d_up)

                # head
                cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)

                # pred
                conf_pred = self.conf_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)
        
                # generate anchors
                fmp_size = conf_pred.shape[-2:]
                anchors = self.generate_anchors(fmp_size, self.stride[level])

                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

                # decode box: [M, 4]
                box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])

                all_conf_preds.append(conf_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
            
            # output dict
            outputs = {"pred_conf": all_conf_preds,       # List(Tensor) [B, M, 1]
                       "pred_cls": all_cls_preds,         # List(Tensor) [B, M, C]
                       "pred_box": all_box_preds,         # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,            # List(Tensor) [B, M, 2]
                       "strides": self.stride}            # List(Int)

            return outputs
