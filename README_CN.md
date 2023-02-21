# YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection
## YOWOv2的网络结构图
![image](./img_files/yowov2.png)


## 配置环境
- 首先，我们建议使用Anaconda来创建一个conda的虚拟环境
```Shell
conda create -n yowo python=3.6
```

- 然后, 请激活已创建的虚拟环境
```Shell
conda activate yowo
```

- 接着，配置环境:
```Shell
pip install -r requirements.txt 
```

项目作者所使用的环境配置:
- PyTorch = 1.9.1
- Torchvision = 0.10.1

为了能够正常运行该项目的代码，请确保您的torch版本为1.x系列。

## 检测结果的可视化图像

![image](./img_files/ucf24_v_Basketball_g07_c04.gif)
![image](./img_files/ucf24_v_Biking_g01_c01.gif)
![image](./img_files/ucf24_v_Fencing_g01_c06.gif)

![image](./img_files/ucf24_v_HorseRiding_g01_c03.gif)
![image](./img_files/ucf24_v_IceDancing_g02_c05.gif)
![image](./img_files/ucf24_v_SalsaSpin_g03_c01.gif)

# 数据集

## UCF101-24:
建议使用者从下面给出的链接来获取 **UCF24** 数据集:

* Google drive

Link: https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?usp=sharing

* 百度网盘

获取链接: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

提取码: hmu6 

## AVA
建议使用这遵从[here](https://github.com/yjh0410/AVA_Dataset)给出的要求来准备 **AVA** 数据集.

# 实验结果
* UCF101-24

|      Model     |  Clip  | GFLOPs |  Params | F-mAP | V-mAP |   FPS   |    Weight    |
|----------------|--------|--------|---------|-------|-------|---------|--------------|
|  YOWOv2-Nano   |   16   |  1.3   | 3.5 M   | 78.8  | 48.0  |   42    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_nano_ucf24.pth) |
|  YOWOv2-Tiny   |   16   |  2.9   | 10.9 M  | 80.5  | 51.3  |   50    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_tiny_ucf24.pth) |
|  YOWOv2-Medium |   16   |  12.0  | 52.0 M  | 83.1  | 50.7  |   42    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_medium_ucf24.pth) |
|  YOWOv2-Large  |   16   |  53.6 | 109.7 M | 85.2  | 52.0  |   30    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_large_ucf24.pth) |
|  YOWOv2-Nano   |   32   |  2.0   | 3.5 M   | 79.4  | 49.0  |   42    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_nano_ucf24_k32.pth) |
|  YOWOv2-Tiny   |   32   |  4.5   | 10.9 M  | 83.0  | 51.2  |   50    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_tiny_ucf24_k32.pth) |
|  YOWOv2-Medium |   32   |  12.7  | 52.0 M  | 83.7  | 52.5  |   40    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_medium_ucf24_k32.pth) |
|  YOWOv2-Large  |   32   |  91.9  | 109.7 M | 87.0  | 52.8  |   22    | [ckpt](https://github.com/yjh0410/YOWOv2/releases/download/yowo_v2_weight/yowo_v2_large_ucf24_k32.pth) |

* 我们使用包含16或32帧（每帧图像的尺寸为224×224）的视频片段来测试模型的FLOPs和FPS。测试FPS时，我们在一张3090GPU上以batch size=1的条件下去完成的，并且，FPS的测试范围包括模型前向推理、后处理以及NMS操作。*

**UCF101-24的检测结果的可视化图像**
![image](./img_files/vis_ucf24.png)


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

**AVA的检测结果的可视化图像**
![image](./img_files/vis_ava.png)


## 训练 YOWOv2
* UCF101-24

使用者可以参考下面的命令来使用 UCF101-24 数据集训练 YOWOv2:

```Shell
python train.py --cuda -d ucf24 --root path/to/dataset -v yowo_v2_nano --num_workers 4 --eval_epoch 1 --max_epoch 8 --lr_epoch 2 3 4 5 -lr 0.0001 -ldr 0.5 -bs 8 -accu 16 -K 16
```

或者，使用者可以运行已准备好的脚本来训练。

```Shell
sh train_ucf.sh
```

为了顺利运行该脚本，请使用者根据自己的本地设备的情况来修改其中的参数。

* AVA

使用者可以参考下面的命令来使用 AVA 数据集训练YOWOv2:


```Shell
python train.py --cuda -d ava_v2.2 --root path/to/dataset -v yowo_v2_nano --num_workers 4 --eval_epoch 1 --max_epoch 10 --lr_epoch 3 4 5 6 -lr 0.0001 -ldr 0.5 -bs 8 -accu 16 -K 16 --eval
```

或者，使用者可以运行已准备好的脚本来训练。

```Shell
sh train_ava.sh
```

为了顺利运行该脚本，请使用者根据自己的本地设备的情况来修改其中的参数。

##  测试 YOWOv2
* UCF101-24
使用者可以参考下面的命令来在 UCF101-24 数据集测试YOWOv2:

```Shell
python test.py --cuda -d ucf24 -v yowo_v2_nano --weight path/to/weight -size 224 --show
```

* AVA
使用者可以参考下面的命令来使用 AVA 数据集测试YOWOv2:

```Shell
python test.py --cuda -d ava_v2.2 -v yowo_v2_nano --weight path/to/weight -size 224 --show
```

##  使用 AVA 的视频来测试 YOWOv2
使用者可以参考下面的命令来使用 AVA 的视频测试YOWOv2:

```Shell
python test_video_ava.py --cuda -d ava_v2.2 -v yowo_v2_nano --weight path/to/weight --video path/to/video --show
```

注意，使用者需要将 ```path/to/video``` 修改为要测试的视频的文件路径。

## 验证 YOWOv2
* UCF101-24
使用者可以参考下面的命令来使用 UCF101-24 验证YOWOv2的性能:

```Shell
# 计算 Frame mAP
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
# 计算 Video mAP
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
使用者可以参考下面的命令来使用 AVA 数据集验证YOWOv2的性能：

```Shell
python eval.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo_v2_nano \
        -bs 16 \
        --weight path/to/weight
```

在AVA数据集上，我们仅计算`Frame mAP@0.5 IoU`指标。

## Demo
使用者可以参考下面的命令来测试本地的视频文件：

```Shell
# run demo
python demo.py --cuda -d ucf24 -v yowo_v2_nano -size 224 --weight path/to/weight --video path/to/video --show
                      -d ava_v2.2
```

注意，使用者需要将 ```path/to/video``` 修改为要测试的视频的文件路径。

**真实场景下的一些检测结果的可视化图像**
![image](./img_files/vis_demo.png)


## 参考文献
如果你正在使用我们的代码，请引用我们的论文：

```
@article{yang2023yowov2,
  title={YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection},
  author={Yang, Jianhua and Kun, Dai},
  journal={arXiv preprint arXiv:2302.06848},
  year={2023}
}
```
