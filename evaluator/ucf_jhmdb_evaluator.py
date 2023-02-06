import os
import torch
import numpy as np
from scipy.io import loadmat

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset, UCF_JHMDB_VIDEO_Dataset
from utils.box_ops import rescale_bboxes

from .cal_frame_mAP import evaluate_frameAP
from .cal_video_mAP import evaluate_videoAP


class UCF_JHMDB_Evaluator(object):
    def __init__(self,
                 data_root=None,
                 dataset='ucf24',
                 model_name='yowo',
                 metric='fmap',
                 img_size=224,
                 len_clip=1,
                 batch_size=1,
                 conf_thresh=0.01,
                 iou_thresh=0.5,
                 transform=None,
                 collate_fn=None,
                 gt_folder=None,
                 save_path=None):
        self.data_root = data_root
        self.dataset = dataset
        self.model_name = model_name
        self.img_size = img_size
        self.len_clip = len_clip
        self.batch_size = batch_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.collate_fn = collate_fn

        self.gt_folder = gt_folder
        self.save_path = save_path

        self.gt_file = os.path.join(data_root, 'splitfiles/finalAnnots.mat')
        self.testlist = os.path.join(data_root, 'splitfiles/testlist01.txt')

        # dataset
        if metric == 'fmap':
            self.testset = UCF_JHMDB_Dataset(
                data_root=data_root,
                dataset=dataset,
                img_size=img_size,
                transform=transform,
                is_train=False,
                len_clip=len_clip,
                sampling_rate=1)
            self.num_classes = self.testset.num_classes
        elif metric == 'vmap':
            self.testset = UCF_JHMDB_VIDEO_Dataset(
                data_root=data_root,
                dataset=dataset,
                img_size=img_size,
                transform=transform,
                len_clip=len_clip,
                sampling_rate=1)
            self.num_classes = self.testset.num_classes


    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):
        print("Metric: Frame mAP")
        # dataloader
        self.testloader = torch.utils.data.DataLoader(
            dataset=self.testset, 
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn, 
            num_workers=4,
            drop_last=False,
            pin_memory=True
            )
        
        epoch_size = len(self.testloader)

        # inference
        for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(self.testloader):
            # to device
            batch_video_clip = batch_video_clip.to(model.device)

            with torch.no_grad():
                # inference
                batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

                # process batch
                for bi in range(len(batch_scores)):
                    frame_id = batch_frame_id[bi]
                    scores = batch_scores[bi]
                    labels = batch_labels[bi]
                    bboxes = batch_bboxes[bi]
                    target = batch_target[bi]

                    # rescale bbox
                    orig_size = target['orig_size']
                    bboxes = rescale_bboxes(bboxes, orig_size)

                    if not os.path.exists('results'):
                        os.mkdir('results')

                    if self.dataset == 'ucf24':
                        detection_path = os.path.join('results', 'ucf_detections', self.model_name, 'detections_' + str(epoch), frame_id)
                        current_dir = os.path.join('results', 'ucf_detections',  self.model_name, 'detections_' + str(epoch))
                        if not os.path.exists('results/ucf_detections/'):
                            os.mkdir('results/ucf_detections/')
                        if not os.path.exists('results/ucf_detections/'+self.model_name):
                            os.mkdir('results/ucf_detections/'+self.model_name)
                        if not os.path.exists(current_dir):
                            os.mkdir(current_dir)
                    else:
                        detection_path = os.path.join('results', 'jhmdb_detections',  self.model_name, 'detections_' + str(epoch), frame_id)
                        current_dir = os.path.join('results', 'jhmdb_detections',  self.model_name, 'detections_' + str(epoch))
                        if not os.path.exists('results/jhmdb_detections/'):
                            os.mkdir('results/jhmdb_detections/')
                        if not os.path.exists('results/jhmdb_detections/'+self.model_name):
                            os.mkdir('results/jhmdb_detections/'+self.model_name)
                        if not os.path.exists(current_dir):
                            os.mkdir(current_dir)

                    with open(detection_path, 'w+') as f_detect:
                        for score, label, bbox in zip(scores, labels, bboxes):
                            x1 = round(bbox[0])
                            y1 = round(bbox[1])
                            x2 = round(bbox[2])
                            y2 = round(bbox[3])
                            cls_id = int(label) + 1

                            f_detect.write(
                                str(cls_id) + ' ' + str(score) + ' ' \
                                    + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

                if iter_i % 100 == 0:
                    log_info = "[%d / %d]" % (iter_i, epoch_size)
                    print(log_info, flush=True)

        print('calculating Frame mAP ...')
        metric_list = evaluate_frameAP(self.gt_folder, current_dir, self.iou_thresh,
                              self.save_path, self.dataset, show_pr_curve)
        for metric in metric_list:
            print(metric)


    def evaluate_video_map(self, model):
        print("Metric: Video mAP")
        video_testlist = []
        with open(self.testlist, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                video_testlist.append(line)

        detected_boxes = {}
        gt_videos = {}

        gt_data = loadmat(self.gt_file)['annot']
        n_videos = gt_data.shape[1]
        print('loading gt tubes ...')
        for i in range(n_videos):
            video_name = gt_data[0][i][1][0]
            if video_name in video_testlist:
                n_tubes = len(gt_data[0][i][2][0])
                v_annotation = {}
                all_gt_boxes = []
                for j in range(n_tubes):  
                    gt_one_tube = [] 
                    tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                    tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                    tube_class = gt_data[0][i][2][0][j][2][0][0]
                    tube_data = gt_data[0][i][2][0][j][3]
                    tube_length = tube_end_frame - tube_start_frame + 1
                
                    for k in range(tube_length):
                        gt_boxes = []
                        gt_boxes.append(int(tube_start_frame+k))
                        gt_boxes.append(float(tube_data[k][0]))
                        gt_boxes.append(float(tube_data[k][1]))
                        gt_boxes.append(float(tube_data[k][0]) + float(tube_data[k][2]))
                        gt_boxes.append(float(tube_data[k][1]) + float(tube_data[k][3]))
                        gt_one_tube.append(gt_boxes)
                    all_gt_boxes.append(gt_one_tube)

                v_annotation['gt_classes'] = tube_class
                v_annotation['tubes'] = np.array(all_gt_boxes)
                gt_videos[video_name] = v_annotation

        # inference
        print('inference ...')
        for i, line in enumerate(lines):
            line = line.rstrip()
            if i % 50 == 0:
                print('Video: [%d / %d] - %s' % (i, len(lines), line))
            
            # set video
            self.testset.set_video_data(line)

            # dataloader
            self.testloader = torch.utils.data.DataLoader(
                dataset=self.testset, 
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn, 
                num_workers=4,
                drop_last=False,
                pin_memory=True
                )
                    
            for iter_i, (batch_img_name, batch_video_clip, batch_target) in enumerate(self.testloader):
                # to device
                batch_video_clip = batch_video_clip.to(model.device)

                with torch.no_grad():
                    # inference
                    batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

                    # process batch
                    for bi in range(len(batch_scores)):
                        img_name = batch_img_name[bi]
                        scores = batch_scores[bi]
                        labels = batch_labels[bi]
                        bboxes = batch_bboxes[bi]
                        target = batch_target[bi]

                        # rescale bbox
                        orig_size = target['orig_size']
                        bboxes = rescale_bboxes(bboxes, orig_size)

                        img_annotation = {}
                        for cls_idx in range(self.num_classes):
                            inds = np.where(labels == cls_idx)[0]
                            c_bboxes = bboxes[inds]
                            c_scores = scores[inds]
                            # [n_box, 5]
                            boxes = np.concatenate([c_bboxes, c_scores[..., None]], axis=-1)
                            img_annotation[cls_idx+1] = boxes
                        detected_boxes[img_name] = img_annotation

            # delete testloader
            del self.testloader

        iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
        print('calculating video mAP ...')
        for iou_th in iou_list:
            per_ap = evaluate_videoAP(gt_videos, detected_boxes, self.num_classes, iou_th, True)
            video_mAP = sum(per_ap) / len(per_ap)
            print('-------------------------------')
            print('V-mAP @ {} IoU:'.format(iou_th))
            print('--Per AP: ', per_ap)
            print('--mAP: ', round(video_mAP, 2))


if __name__ == "__main__":
    pass