#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import csv
from collections import defaultdict


logger = logging.getLogger(__name__)
FPS = 30
AVA_VALID_FRAMES = range(902, 1799)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.
    Args:
      exclusions_file: A file object containing a csv of video-id,timestamp.
    Returns:
      A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
      or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        with open(exclusions_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 2, "Expected only 2 columns, got: " + row
                excluded.add(make_image_key(row[0], row[1]))
    return excluded


def load_image_lists(frames_dir, frame_list, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        frames_dir (str): path to frames dir.
        frame_list (str): path to frame list.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    # frame_list_dir is /data3/ava/frame_lists/
    # contains 'train.csv' and 'val.csv'
    if is_train:
        list_name = "train.csv"
    else:
        list_name = "val.csv"

    list_filename = os.path.join(frame_list, list_name)

    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    with open(list_filename, "r") as f:
        f.readline()
        for line in f:
            row = line.split()
            # The format of each row should follow:
            # original_vido_id video_id frame_id path labels.
            assert len(row) == 5
            video_name = row[0]

            if video_name not in video_name_to_idx:
                idx = len(video_name_to_idx)
                video_name_to_idx[video_name] = idx
                video_idx_to_name.append(video_name)

            data_key = video_name_to_idx[video_name]

            image_paths[data_key].append(os.path.join(frames_dir, row[3]))

    image_paths = [image_paths[i] for i in range(len(image_paths))]

    print("Finished loading image paths from: {}".format(list_filename))

    return image_paths, video_idx_to_name


def load_boxes_and_labels(gt_box_list, exclusion_file, is_train=False, full_test_on_val=False):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    ann_filename = gt_box_list
    all_boxes = {}
    count = 0
    unique_box_count = 0
    excluded_keys = read_exclusions(exclusion_file)

    with open(ann_filename, 'r') as f:
        for line in f:
            row = line.strip().split(',')

            video_name, frame_sec = row[0], int(row[1])
            key = "%s,%04d" % (video_name, frame_sec)
            # if mode == 'train' and key in excluded_keys:
            if key in excluded_keys:
                print("Found {} to be excluded...".format(key))
                continue

            # Only select frame_sec % 4 = 0 samples for validation if not
            # set FULL_TEST_ON_VAL (default False)
            if not is_train and not full_test_on_val and frame_sec % 4 != 0:
                continue
            # Box with [x1, y1, x2, y2] with a range of [0, 1] as float
            box_key = ",".join(row[2:6])
            box = list(map(float, row[2:6]))
            label = -1 if row[6] == "" else int(row[6])
            if video_name not in all_boxes:
                all_boxes[video_name] = {}
                for sec in AVA_VALID_FRAMES:
                    all_boxes[video_name][sec] = {}
            if box_key not in all_boxes[video_name][frame_sec]:
                all_boxes[video_name][frame_sec][box_key] = [box, []]
                unique_box_count += 1

            all_boxes[video_name][frame_sec][box_key][1].append(label)
            if label != -1:
                count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    print("Finished loading annotations from: %s" % ", ".join([ann_filename]))
    print("Number of unique boxes: %d" % unique_box_count)
    print("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        return (sec - 900) * FPS

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            if sec not in AVA_VALID_FRAMES:
                continue

            if len(boxes_and_labels[video_idx][sec]) > 0:
                keyframe_indices.append(
                    (video_idx, sec_idx, sec, sec_to_frame(sec))
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count


def get_max_objs(keyframe_indices, keyframe_boxes_and_labels):
    # max_objs = 0
    # for video_idx, sec_idx, _, _ in keyframe_indices:
    #     num_boxes = len(keyframe_boxes_and_labels[video_idx][sec_idx])
    #     if num_boxes > max_objs:
    #         max_objs = num_boxes

    # return max_objs
    return 50 #### MODIFICATION FOR NOW! TODO: FIX LATER!
