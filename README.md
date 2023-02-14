# YOWOv2

## Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yowo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yowo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

## Visualization

![image](./img_files/ucf24_v_Basketball_g07_c04.gif)
![image](./img_files/ucf24_v_Biking_g01_c01.gif)
![image](./img_files/ucf24_v_Fencing_g01_c06.gif)

![image](./img_files/ucf24_v_HorseRiding_g01_c03.gif)
![image](./img_files/ucf24_v_IceDancing_g02_c05.gif)
![image](./img_files/ucf24_v_SalsaSpin_g03_c01.gif)

# Dataset
You can download **UCF24** from the following links:

## UCF101-24:
* Google drive

Link: https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?usp=sharing

* BaiduYun Disk

Link: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

Password: hmu6 

## AVA
You can use instructions from [here](https://github.com/yjh0410/AVA_Dataset) to prepare **AVA** dataset.

# Experiment
* UCF101-24

|      Model     |  Clip  | GFLOPs |  Params | F-mAP | V-mAP |   FPS   |    Weight    |
|----------------|--------|--------|---------|-------|-------|---------|--------------|
|  YOWOv2-Nano   |   16   |  1.3   | 3.5 M   | 78.8  | 48.0  |   42    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_nano_ucf24.pth) |
|  YOWOv2-Tiny   |   16   |  2.9   | 10.9 M  | 80.5  | 51.3  |   50    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_tiny_ucf24.pth) |
|  YOWOv2-Medium |   16   |  12.0  | 52.0 M  | 83.1  | 50.7  |   42    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_medium_ucf24.pth) |
|  YOWOv2-Large  |   16   |  53.6 | 109.7 M | 85.2  | 52.0  |   30    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_large_ucf24.pth) |
|  YOWOv2-Nano   |   32   |  2.0   | 3.5 M   | 79.4  | 49.0  |   42    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_nano_ucf24_k32.pth) |
|  YOWOv2-Tiny   |   32   |  4.0   | 10.9 M  | 83.0  | 51.2  |   50    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_tiny_ucf24_k32.pth) |
|  YOWOv2-Medium |   32   |  12.7  | 52.0 M  | 83.7  | 52.5  |   40    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_medium_ucf24_k32.pth) |
|  YOWOv2-Large  |   32   |  91.9  | 109.7 M | 87.0  | 52.8  |   22    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_large_ucf24_k32.pth) |

* AVA v2.2

|     Model      |    Clip    |    mAP    |   FPS   |    weight    |
|----------------|------------|-----------|---------|--------------|
|  YOWOv2-Nano   |     16     |   12.6    |   40    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_nano_ava.pth) |
|  YOWOv2-Tiny   |     16     |   14.9    |   49    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_tiny_ava.pth) |
|  YOWOv2-Medium |     16     |   18.4    |   41    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_medium_ava.pth) |
|  YOWOv2-Large  |     16     |   20.2    |   29    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_large_ava.pth) |
|  YOWOv2-Nano   |     32     |   12.7    |   40    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_nano_ucf24_k32.pth) |
|  YOWOv2-Tiny   |     32     |   15.6    |   49    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_tiny_ava_k32.pth) |
|  YOWOv2-Medium |     32     |   18.4    |   40    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_medium_ava_k32.pth) |
|  YOWOv2-Large  |     32     |   21.7    |   22    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_large_ava_k32.pth) |


## Train YOWOv2
* UCF101-24

```Shell
python train.py --cuda -d ucf24 --root path/to/dataset -v yowo_v2_nano --num_workers 4 --eval_epoch 1 --max_epoch 8 --lr_epoch 2 3 4 5 --lr 0.0001 -ldr 0.5 -bs 8 -accu 16
```

or you can just run the script:

```Shell
sh train_ucf.sh
```

* AVA
```Shell
python train.py --cuda -d ava_v2.2 --root path/to/dataset -v yowo_v2_nano --num_workers 4 --eval_epoch 1 --max_epoch 10 --lr_epoch 3 4 5 6 --lr 0.0001 -ldr 0.5 -bs 8 -accu 16 --eval
```

or you can just run the script:

```Shell
sh train_ava.sh
```

##  Test YOWOv2
* UCF101-24
For example:

```Shell
python test.py --cuda -d ucf24 -v yowo_v2_nano --weight path/to/weight -size 224 --show
```

* AVA
For example:

```Shell
python test.py --cuda -d ava_v2.2 -v yowo_v2_nano --weight path/to/weight -size 224 --show
```

##  Test YOWOv2 on AVA video
For example:

```Shell
python test_video_ava.py --cuda -d ava_v2.2 -v yowo_v2_nano --weight path/to/weight --video path/to/video --show
```

Note that you can set ```path/to/video``` to other videos in your local device, not AVA videos.

## Evaluate YOWOv2
* UCF101-24
For example:

```Shell
# Frame mAP
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_nano \
        -bs 16 \
        -size 224 \
        --weight path/to/weight \
        --cal_frame_mAP \
```

```Shell
# Video mAP
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_nano \
        -bs 16 \
        -size 224 \
        --weight path/to/weight \
        --cal_video_mAP \
```

* AVA

Run the following command to calculate frame mAP@0.5 IoU:

```Shell
python eval.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo_v2_nano \
        -bs 16 \
        --weight path/to/weight
```

## Demo
```Shell
# run demo
python demo.py --cuda -d ucf24 -v yowo_v2_nano -size 224 --weight path/to/weight --video path/to/video --show
                      -d ava_v2.2
```

## References
If you are using our code, please consider citing our paper.

Comming soon ...
