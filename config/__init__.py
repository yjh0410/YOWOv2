from .dataset_config import dataset_config
from .yowo_v2_config import yowo_v2_config


def build_model_config(args):
    print('==============================')
    print('Model Config: {} '.format(args.version.upper()))
    
    if 'yowo_v2_' in args.version:
        m_cfg = yowo_v2_config[args.version]

    return m_cfg


def build_dataset_config(args):
    print('==============================')
    print('Dataset Config: {} '.format(args.dataset.upper()))
    
    d_cfg = dataset_config[args.dataset]

    return d_cfg
