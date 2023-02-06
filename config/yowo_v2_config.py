# Model configuration


yowo_v2_config = {
    'yowo_v2_nano': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_nano',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '1.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': True,
    },

    'yowo_v2_tiny': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_tiny',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    'yowo_v2_medium': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_large',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '2.0x',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

    'yowo_v2_large': {
        # backbone
        ## 2D
        'backbone_2d': 'yolo_free_large',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'resnext101',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
    },

}