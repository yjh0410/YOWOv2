from .yowo.build import build_yowo


def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    # build action detector
    if 'yowo_v2_' in args.version:
        model, criterion = build_yowo(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            resume=resume
            )

    return model, criterion

