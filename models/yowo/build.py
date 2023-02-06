import torch
from .yowo import YOWO
from .loss import build_criterion


# build YOWO detector
def build_yowo(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # build YOWO
    model = YOWO(
        cfg = m_cfg,
        device = device,
        num_classes = num_classes,
        conf_thresh = args.conf_thresh,
        nms_thresh = args.nms_thresh,
        topk = args.topk,
        trainable = trainable,
        multi_hot = d_cfg['multi_hot'],
        )

    if trainable:
        # Freeze backbone
        if args.freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if args.freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # keep training       
        if resume is not None:
            print('keep training: ', resume)
            checkpoint = torch.load(resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        # build criterion
        criterion = build_criterion(
            args, d_cfg['train_size'], num_classes, d_cfg['multi_hot'])
    
    else:
        criterion = None
                        
    return model, criterion
