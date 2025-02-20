# Medical-Image-Segmentation


医学图像分割框架

# 目录

[TOC]

## 环境配置

### 代码框架

- [Anaconda](https://www.anaconda.com/products/individual)
- [Pytorch](https://pytorch.org)

### 所需库

```
* python >= 3.8
* cuda >= 11.7
```



## 数据集设置

> images和masks中图片的命名没有要求,只需确保数据集按照以下格式组织即可; 图片格式可以是jpg/png/tif

```
<dataset name>
    |
    ├── train
    |     ├── images
    │     |     ├── xxx.png
    │     |     ├── yyy.png
    │     |     ├── zzz.png
    |     |     ├── ... 
    |     |
    │     └── masks
    |           ├── xxx.png
    |           ├── yyy.png
    |           ├── zzz.png
    |           ├── ... 
    |
    |
    └── test
          ├── images
          |     ├── xxx.png
          |     ├── yyy.png
          |     ├── zzz.png
          |     ├── ... 
          |
          └── masks
                ├── xxx.png
                ├── yyy.png
                ├── zzz.png
                ├── ... 

```

## 运行

设置数据集目录后，代码可以通过以下脚本运行。

### 训练

```bash
python main.py  <--args1 args1_value> <--args2 args2_value>
```

### 测试

```bash
python main.py --phase test <--args1 args1_value> <--args2 args2_value>
```

### 实际用例及公共参数注释

算法链接：[DconnNet](https://arxiv.org/abs/2304.00145) | [LaplacianFormer](https://arxiv.org/abs/2309.00108) | [MISSFormer](https://arxiv.org/abs/2109.07162) | [RollingUNet](https://ojs.aaai.org/index.php/AAAI/article/view/28173) | [UNet](https://arxiv.org/abs/1505.04597)

```bash
python main.py       --epochs               # 训练轮数
                     --batch_size      
                     --metric_list          # 多个指标名称
                     --num_workers
                     --seed                 # 随机种子
                     --output               # 输出存放的位置
                     --dataset              # 数据集名称
                     --print_interval       # 训练终端输出的间隔
                     --save_interval        # 训练保存lr和loss的间隔
                     --weight_decay         # optimizer相关参数
                     --momentum             # optimizer相关参数
                     --lr                   # 学习率
                     --nesterov             # optimizer相关参数
                     --min_lr               # lr_scheduler相关参数
                     --factor               # lr_scheduler相关参数
                     --patience             # lr_scheduler相关参数
                     --milestones           # lr_scheduler相关参数
                     --gamma                # lr_scheduler相关参数
                     --early_stopping       # lr_scheduler相关参数
                     --cfg                  # lr_scheduler相关参数
                     --phase                # train/test
                     --control_print        # 在终端打印结果
                     --control_save         # 保存结果到文件
                     --control_save_end     # 保存最终权重
                     --data_path            # 数据集存放目录
                     --gpu                  # 默认为0
                     --model_name           # 算法名称
                     --img_size             # 图片裁剪大小
                     --load_model           # 下载权重
```

常用参数设置下的运行命令:

Eg. 

训练:

```bash
python main.py --dataset CHASE_DB1 --epochs 200 --phase train --model_name rollingunet --batch_size 2 --lr 1e-4 --img_size 512 --print_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
```

测试:

```bash
python main.py --dataset CHASE_DB1 --epochs 200 --phase test --model_name rollingunet --batch_size 2 --lr 1e-4 --img_size 512 --print_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
```


