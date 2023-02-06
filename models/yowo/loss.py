import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class SigmoidFocalLoss(object):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logits, targets):      
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                        target=targets, 
                                                        reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()

        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class Criterion(object):
    def __init__(self, args, img_size, num_classes=80, multi_hot=False):
        self.num_classes = num_classes
        self.img_size = img_size
        self.loss_conf_weight = args.loss_conf_weight
        self.loss_cls_weight = args.loss_cls_weight
        self.loss_reg_weight = args.loss_reg_weight
        self.focal_loss = args.focal_loss
        self.multi_hot = multi_hot

        # loss
        self.obj_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')
            
        # matcher
        self.matcher = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=args.center_sampling_radius,
            topk_candidate=args.topk_candicate
            )

    def __call__(self, outputs, targets):        
        """
            outputs['pred_conf']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        # preds: [B, M, C]
        conf_preds = torch.cat(outputs['pred_conf'], dim=1)
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # label assignment
        cls_targets = []
        box_targets = []
        conf_targets = []
        fg_masks = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # denormalize tgt_bbox
            tgt_bboxes *= self.img_size

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                # There is no valid gt
                cls_target = conf_preds.new_zeros((0, self.num_classes))
                box_target = conf_preds.new_zeros((0, 4))
                conf_target = conf_preds.new_zeros((num_anchors, 1))
                fg_mask = conf_preds.new_zeros(num_anchors).bool()
            else:
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.matcher(
                    fpn_strides = fpn_strides,
                    anchors = anchors,
                    pred_conf = conf_preds[batch_idx],
                    pred_cls = cls_preds[batch_idx], 
                    pred_box = box_preds[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes,
                    )

                conf_target = fg_mask.unsqueeze(-1)
                box_target = tgt_bboxes[matched_gt_inds]
                if self.multi_hot:
                    cls_target = gt_matched_classes.float()
                else:
                    cls_target = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            conf_targets.append(conf_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        conf_targets = torch.cat(conf_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_foregrounds = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)

        # conf loss
        loss_conf = self.obj_lossf(conf_preds.view(-1, 1), conf_targets.float())
        loss_conf = loss_conf.sum() / num_foregrounds
        
        # cls loss
        matched_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.cls_lossf(matched_cls_preds, cls_targets)
        loss_cls = loss_cls.sum() / num_foregrounds

        # box loss
        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        ious = get_ious(matched_box_preds,
                        box_targets,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_box = (1.0 - ious).sum() / num_foregrounds

        # total loss
        losses = self.loss_conf_weight * loss_conf + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_reg_weight * loss_box

        loss_dict = dict(
                loss_conf = loss_conf,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict


def build_criterion(args, img_size, num_classes, multi_hot=False):
    criterion = Criterion(args, img_size, num_classes, multi_hot)
    
    return criterion
    