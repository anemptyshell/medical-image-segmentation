

## Prerequisites

* torch>=1.4.0




## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 2):

inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```

For binary segmentation problems, just use folder 0.

```

## Train and test

### train
```
> python main.py --model_name 
```

### test
```
> python main.py --model_name 
```

```

# unet
python main.py --dataset isic2017 --epochs 100 --phase train --model_name unet --batch_size 2 --print_interval 40 --save_interval 10 --val_interval 10
python main.py --dataset isic2017 --epochs 200 --phase test --model_name unet --batch_size 2 --print_interval 40 --save_interval 1 --val_interval 10


# missformer
python main.py --dataset isic2017 --epochs 200 --phase train --img_size 224 --model_name missformer --batch_size 2 --optimizer_name SGD --print_interval 40 --save_interval 1 --val_interval 10 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score confusion_matrix

python main.py --dataset isic2017 --epochs 200 --phase test --img_size 224 --model_name missformer --batch_size 2 --print_interval 40 --save_interval 1 --val_interval 10 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score confusion_matrix

--------------------------------------------
python main.py --dataset CHASE_DB1 --epochs 100 --phase train --img_size 224 --model_name missformer --batch_size 2 --print_interval 1 --save_interval 1 --lr 4e-3


### laplacianformer
python main.py --dataset isic2017 --epochs 200 --phase train --img_size 224 --model_name laplacianformer --batch_size 2 --optimizer_name SGD --print_interval 40 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score confusion_matrix 

python main.py --dataset isic2017 --epochs 200 --phase test --img_size 224 --model_name laplacianformer --batch_size 2 --optimizer_name SGD --print_interval 40 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score confusion_matrix 

#### selfregunet
python main.py --dataset isic2017 --epochs 150 --phase train --img_size 224 --model_name selfregunet --batch_size 2 --optimizer_name SGD --print_interval 50 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score confusion_matrix --lr 1e-2


#### dconnnet
python main.py --dataset CHASE_DB1 --epochs 60 --phase train --model_name dconnnet --batch_size 1 --lr 0.001 --gamma 0.5 --folds 5


####rollingunet
python main.py --dataset CHASE_DB1 --epochs 400 --phase test --model_name rollingunet --batch_size 2 --lr 1e-4 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 