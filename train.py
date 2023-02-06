import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import time
import argparse
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from config import build_dataset_config, build_model_config
from models import build_model


GLOBAL_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # Visualization
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='./weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='use tensorboard')

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False, 
                        help='do evaluation during training.')
    parser.add_argument('--eval_epoch', default=1, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--save_dir', default='inference_results/',
                        type=str, help='save inference results.')
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')

    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on a single GPU.')
    parser.add_argument('-tbs', '--test_batch_size', default=16, type=int, 
                        help='test batch size on a single GPU.')
    parser.add_argument('-accu', '--accumulate', default=1, type=int, 
                        help='gradient accumulate.')
    parser.add_argument('-lr', '--base_lr', default=0.0001, type=float, 
                        help='base lr.')
    parser.add_argument('-ldr', '--lr_decay_ratio', default=0.5, type=float, 
                        help='base lr.')

    # Epoch
    parser.add_argument('--max_epoch', default=10, type=int, 
                        help='max epoch.')
    parser.add_argument('--lr_epoch', nargs='+', default=[2,3,4], type=int,
                        help='lr epoch to decay')

    # Model
    parser.add_argument('-v', '--version', default='yowo_v2_tiny', type=str,
                        help='build YOWOv2')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold. We suggest 0.5 for UCF24 and AVA.')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk prediction candidates.')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('--freeze_backbone_2d', action="store_true", default=False,
                        help="freeze 2D backbone.")
    parser.add_argument('--freeze_backbone_3d', action="store_true", default=False,
                        help="freeze 3d backbone.")
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    # Dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, ava_v2.2')
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset/STAD/',
                        help='data root')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')

    # Matcher
    parser.add_argument('--center_sampling_radius', default=2.5, type=float, 
                        help='conf loss weight factor.')
    parser.add_argument('--topk_candicate', default=10, type=int, 
                        help='cls loss weight factor.')

    # Loss
    parser.add_argument('--loss_conf_weight', default=1, type=float, 
                        help='conf loss weight factor.')
    parser.add_argument('--loss_cls_weight', default=1, type=float, 
                        help='cls loss weight factor.')
    parser.add_argument('--loss_reg_weight', default=5, type=float, 
                        help='reg loss weight factor.')
    parser.add_argument('-fl', '--focal_loss', action="store_true", default=False,
                        help="use focal loss for classification.")
    
    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=True)

    # dataloader
    batch_size = args.batch_size * distributed_utils.get_world_size()
    dataloader = build_dataloader(args, dataset, batch_size, CollateFunc(), is_train=True)

    # build model
    model, criterion = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes, 
        trainable=True,
        resume=args.resume
        )
    model = model.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Compute FLOPs and Params
    if distributed_utils.is_main_process():
        model_copy = deepcopy(model_without_ddp)
        FLOPs_and_Params(
            model=model_copy,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            device=device)
        del model_copy

    # optimizer
    base_lr = args.base_lr
    accumulate = args.accumulate
    optimizer, start_epoch = build_optimizer(d_cfg, model_without_ddp, base_lr, args.resume)

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_epoch, args.lr_decay_ratio)

    # warmup scheduler
    warmup_scheduler = build_warmup(d_cfg, base_lr=base_lr)

    # training configuration
    max_epoch = args.max_epoch
    epoch_size = len(dataloader)
    warmup = True

    # eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        eval_one_epoch(args, model_without_ddp, evaluator, 0, path_to_save)

    # start to train
    t0 = time.time()
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size

            # warmup
            if ni < d_cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == d_cfg['wp_iter'] and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, lr=base_lr, base_lr=base_lr)

            # to device
            video_clips = video_clips.to(device)

            # inference
            outputs = model(video_clips)
            
            # loss
            loss_dict = criterion(outputs, targets)
            losses = loss_dict['losses']

            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward
            losses /= accumulate
            losses.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                    
            # Display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                print_log(cur_lr, epoch,  max_epoch, iter_i, epoch_size,loss_dict_reduced, t1-t0, accumulate)
            
                t0 = time.time()

        lr_scheduler.step()
        
        # evaluation
        if epoch % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            eval_one_epoch(args, model_without_ddp, evaluator, epoch, path_to_save)


def eval_one_epoch(args, model_eval, evaluator, epoch, path_to_save):
    # check evaluator
    if distributed_utils.is_main_process():
        if evaluator is None:
            print('No evaluator ... save model and go on training.')
            
        else:
            print('eval ...')
            # set eval mode
            model_eval.trainable = False
            model_eval.eval()

            # evaluate
            evaluator.evaluate_frame_map(model_eval, epoch + 1)
                
            # set train mode.
            model_eval.trainable = True
            model_eval.train()

        # save model
        print('Saving state, epoch:', epoch + 1)
        weight_name = '{}_epoch_{}.pth'.format(args.version, epoch+1)
        checkpoint_path = os.path.join(path_to_save, weight_name)
        torch.save({'model': model_eval.state_dict(),
                    'epoch': epoch,
                    'args': args}, 
                    checkpoint_path)                      

    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()


def print_log(lr, epoch, max_epoch, iter_i, epoch_size, loss_dict, time, accumulate):
    # basic infor
    log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
    log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
    log += '[lr: {:.6f}]'.format(lr[0])
    # loss infor
    for k in loss_dict.keys():
        if k == 'losses':
            log += '[{}: {:.2f}]'.format(k, loss_dict[k] * accumulate)
        else:
            log += '[{}: {:.2f}]'.format(k, loss_dict[k])

    # other infor
    log += '[time: {:.2f}]'.format(time)

    # print log infor
    print(log, flush=True)


if __name__ == '__main__':
    train()
