import os

import torch
import torch.nn as nn

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.ava import AVA_Dataset
from dataset.transforms import Augmentation, BaseTransform

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator


def build_dataset(d_cfg, args, is_train=False):
    """
        d_cfg: dataset config
    """
    # transform
    augmentation = Augmentation(
        img_size=d_cfg['train_size'],
        jitter=d_cfg['jitter'],
        hue=d_cfg['hue'],
        saturation=d_cfg['saturation'],
        exposure=d_cfg['exposure']
        )
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        )

    # dataset
    if args.dataset in ['ucf24', 'jhmdb21']:
        data_dir = os.path.join(args.root, 'ucf24')

        # dataset
        dataset = UCF_JHMDB_Dataset(
            data_root=data_dir,
            dataset=args.dataset,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            is_train=is_train,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
            )
        num_classes = dataset.num_classes

        # evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            batch_size=args.test_batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            gt_folder=d_cfg['gt_folder'],
            save_path='./evaluator/eval_results/',
            transform=basetransform,
            collate_fn=CollateFunc()            
        )

    elif args.dataset == 'ava_v2.2':
        data_dir = os.path.join(args.root, 'AVA_Dataset')
        
        # dataset
        dataset = AVA_Dataset(
            cfg=d_cfg,
            data_root=data_dir,
            is_train=True,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
        )
        num_classes = 80

        # evaluator
        evaluator = AVA_Evaluator(
            d_cfg=d_cfg,
            data_root=data_dir,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate'],
            batch_size=args.test_batch_size,
            transform=basetransform,
            collate_fn=CollateFunc(),
            full_test_on_val=False,
            version='v2.2'
            )

    else:
        print('unknow dataset !! Only support ucf24 & jhmdb21 & ava_v2.2 !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    if not args.eval:
        # no evaluator during training stage
        evaluator = None

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, batch_size, collate_fn=None, is_train=False):
    if is_train:
        # distributed
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                            batch_size, 
                                                            drop_last=True)
        # train dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            pin_memory=True
            )
    else:
        # test dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True
            )
    
    return dataloader
    

def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class CollateFunc(object):
    def __call__(self, batch):
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_key_target
