#!/usr/bin/python
# encoding: utf-8

import os
import random
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image


# Dataset for UCF24 & JHMDB
class UCF_JHMDB_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 is_train=False,
                 len_clip=16,
                 sampling_rate=1):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
            
        if self.is_train:
            self.split_list = 'trainlist.txt'
        else:
            self.split_list = 'testlist.txt'

        # load data
        with open(os.path.join(data_root, self.split_list), 'r') as file:
            self.file_names = file.readlines()
        self.num_samples  = len(self.file_names)

        if dataset == 'ucf24':
            self.num_classes = 24
        elif dataset == 'jhmdb21':
            self.num_classes = 21


    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):
        # load a data
        frame_idx, video_clip, target = self.pull_item(index)

        return frame_idx, video_clip, target


    def pull_item(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        image_path = self.file_names[index].rstrip()

        img_split = image_path.split('/')  # ex. ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # image name
        img_id = int(img_split[-1][:5])

        # path to label
        label_path = os.path.join(self.data_root, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))

        # image folder
        img_folder = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2])

        # frame numbers
        if self.dataset == 'ucf24':
            max_num = len(os.listdir(img_folder))
        elif self.dataset == 'jhmdb21':
            max_num = len(os.listdir(img_folder)) - 1

        # sampling rate
        if self.is_train:
            d = random.randint(1, 2)
        else:
            d = self.sampling_rate

        # load images
        video_clip = []
        for i in reversed(range(self.len_clip)):
            # make it as a loop
            img_id_temp = img_id - i * d
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            if self.dataset == 'ucf24':
                path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2] ,'{:05d}.jpg'.format(img_id_temp))
            elif self.dataset == 'jhmdb21':
                path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2] ,'{:05d}.png'.format(img_id_temp))
            frame = Image.open(path_tmp).convert('RGB')
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

            frame_id = img_split[1] + '_' +img_split[2] + '_' + img_split[3]

        # load an annotation
        if os.path.getsize(label_path):
            target = np.loadtxt(label_path)
        else:
            target = None

        # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]
        label = target[..., :1]
        boxes = target[..., 1:]
        target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)
            
        # transform
        video_clip, target = self.transform(video_clip, target)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)

        # reformat target
        target = {
            'boxes': target[:, :4].float(),      # [N, 4]
            'labels': target[:, -1].long() - 1,    # [N,]
            'orig_size': [ow, oh],
            'video_idx':frame_id[:-10]
        }

        return frame_id, video_clip, target


    def pull_anno(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        image_path = self.file_names[index].rstrip()

        img_split = image_path.split('/')  # ex. ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # image name
        img_id = int(img_split[-1][:5])

        # path to label
        label_path = os.path.join(self.data_root, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))

        # load an annotation
        target = np.loadtxt(label_path)
        target = target.reshape(-1, 5)

        return target
        

# Video Dataset for UCF24 & JHMDB
class UCF_JHMDB_VIDEO_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 len_clip=16,
                 sampling_rate=1):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
            
        if dataset == 'ucf24':
            self.num_classes = 24
        elif dataset == 'jhmdb21':
            self.num_classes = 21


    def set_video_data(self, line):
        self.line = line

        # load a video
        self.img_folder = os.path.join(self.data_root, 'rgb-images', self.line)

        if self.dataset == 'ucf24':
            self.label_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))
        elif self.dataset == 'jhmdb21':
            self.label_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.png')))


    def __len__(self):
        return len(self.label_paths)


    def __getitem__(self, index):
        return self.pull_item(index)


    def pull_item(self, index):
        image_path = self.label_paths[index]

        video_split = self.line.split('/')
        video_class = video_split[0]
        video_file = video_split[1]
        # for windows:
        # img_split = image_path.split('\\')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # for linux
        img_split = image_path.split('/')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']

        # image name
        img_id = int(img_split[-1][:5])
        max_num = len(os.listdir(self.img_folder))
        if self.dataset == 'ucf24':
            img_name = os.path.join(video_class, video_file, '{:05d}.jpg'.format(img_id))
        elif self.dataset == 'jhmdb21':
            img_name = os.path.join(video_class, video_file, '{:05d}.png'.format(img_id))

        # load video clip
        video_clip = []
        for i in reversed(range(self.len_clip)):
            # make it as a loop
            img_id_temp = img_id - i
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            if self.dataset == 'ucf24':
                path_tmp = os.path.join(self.data_root, 'rgb-images', video_class, video_file ,'{:05d}.jpg'.format(img_id_temp))
            elif self.dataset == 'jhmdb21':
                path_tmp = os.path.join(self.data_root, 'rgb-images', video_class, video_file ,'{:05d}.png'.format(img_id_temp))
            frame = Image.open(path_tmp).convert('RGB')
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

        # transform
        video_clip, _ = self.transform(video_clip, normalize=False)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)
        orig_size = [ow, oh]  # width, height

        target = {'orig_size': [ow, oh]}

        return img_name, video_clip, target




if __name__ == '__main__':
    import cv2
    from transforms import Augmentation, BaseTransform

    data_root = 'D:/python_work/spatial-temporal_action_detection/dataset/ucf24'
    dataset = 'ucf24'
    is_train = True
    img_size = 224
    len_clip = 16
    trans_config = {
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5
    }
    train_transform = Augmentation(
        img_size=img_size,
        jitter=trans_config['jitter'],
        saturation=trans_config['saturation'],
        exposure=trans_config['exposure']
        )
    val_transform = BaseTransform(img_size=img_size)

    train_dataset = UCF_JHMDB_Dataset(
        data_root=data_root,
        dataset=dataset,
        img_size=img_size,
        transform=train_transform,
        is_train=is_train,
        len_clip=len_clip,
        sampling_rate=1
    )

    print(len(train_dataset))
    for i in range(len(train_dataset)):
        frame_id, video_clip, target = train_dataset[i]
        key_frame = video_clip[:, -1, :, :]

        # to numpy
        key_frame = key_frame.permute(1, 2, 0).numpy()
        key_frame = key_frame.astype(np.uint8)

        # to BGR
        key_frame = key_frame[..., (2, 1, 0)]
        H, W, C = key_frame.shape

        key_frame = key_frame.copy()
        bboxes = target['boxes']
        labels = target['labels']

        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            key_frame = cv2.rectangle(key_frame, (x1, y1), (x2, y2), (255, 0, 0))

        # cv2 show
        cv2.imshow('key frame', key_frame)
        cv2.waitKey(0)
        
