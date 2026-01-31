#!/bin/sh

conda activate MIS

cd ~/mis-ft/medical-image-segmentation

# python main.py --dataset STARE --epochs 200 --phase test --img_size 512 --model_name laplacianformer --batch_size 2 --optimizer_name SGD --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score --lr 0.01
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 512 --model_name laplacianformer --batch_size 2 --optimizer_name SGD --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score --lr 0.01
# python main.py --dataset ARIA --epochs 200 --phase test --img_size 512 --model_name laplacianformer --batch_size 2 --optimizer_name SGD --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score --lr 0.01

# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 224 --model_name missformer --batch_size 2 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
# python main.py --dataset ARIA --epochs 200 --phase test --img_size 224 --model_name missformer --batch_size 2 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
# python main.py --dataset STARE --epochs 200 --phase test --img_size 224 --model_name missformer --batch_size 2 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 

## dconn 的 DRIVE还没跑--跑了
# python main.py --dataset CHASE_DB1 --epochs 80 --phase train --model_name dconnnet --batch_size 2 --lr 0.00085 --gamma 0.5 --img_size 960 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 80 --phase test --model_name dconnnet --batch_size 2 --lr 0.00085 --gamma 0.5 --img_size 512 --print_interval 1 --save_interval 1
# python main.py --dataset ARIA --epochs 80 --phase test --model_name dconnnet --batch_size 2 --lr 0.00085 --gamma 0.5 --img_size 512 --print_interval 1 --save_interval 1


## cgma Number of parameter: 30.96M
# python main.py --dataset CHASE_DB1 --epochs 250 --phase test --model_name cgmanet --batch_size 2 --lr 0.005 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset DRIVE --epochs 250 --phase test --model_name cgmanet --batch_size 2 --lr 0.005 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset STARE --epochs 150 --phase train --model_name cgmanet --batch_size 2 --lr 0.01 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset ARIA --epochs 250 --phase test --model_name cgmanet --batch_size 2 --lr 0.005 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score



# ## fang  for edge/skeleton
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 512 --model_name fang --batch_size 2 --optimizer_name SGD --print_interval 1 --save_interval 1  --lr 0.01 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 512 --model_name fang --batch_size 2 --optimizer_name SGD --print_interval 1 --save_interval 1  --lr 0.001 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset STARE --epochs 200 --phase test --img_size 512 --model_name fang --batch_size 2 --optimizer_name SGD --print_interval 1 --save_interval 1  --lr 0.005 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score


# # ### final metric 0.5 0.5
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --model_name rollingunet --batch_size 2 --lr 1e-4 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset DRIVE --epochs 200 --phase test --model_name rollingunet --batch_size 2 --lr 1e-4 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset STARE --epochs 200 --phase test --model_name rollingunet --batch_size 2 --lr 1e-4 --img_size 512 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset DEEPCRACK --epochs 200 --phase test --model_name rollingunet --batch_size 2 --lr 0.0005 --img_size 256 --print_interval 10 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score

# python main.py --dataset CHASE_DB1 --epochs 2 --phase train --img_size 512 --model_name unet --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
# python main.py --dataset STARE --epochs 200 --phase test --img_size 512 --model_name unet --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 512 --model_name unet --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
# python main.py --dataset DEEPCRACK --epochs 200 --phase test --img_size 256 --model_name unet --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 10 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score 
## 统一学习率优化器
python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet --batch_size 2 --optimizer_name Adam --lr 0.0085 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE 
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet --batch_size 2 --optimizer_name Adam --lr 0.0085 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE 
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet --batch_size 2 --optimizer_name Adam --lr 0.0085 --print_interval 1 --save_interval 1 --metric_list DSC ACC SEN SPE IoU PRE 


# #  # ## fang  for edge/skeleton
# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name fang1 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 
# python main.py --dataset DRIVE --epochs 200 --phase train --img_size 256 --model_name fang1 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 
# python main.py --dataset STARE --epochs 200 --phase train --img_size 256 --model_name fang1 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 

# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name fang1 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name fang1 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name fang1 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 


# python main.py --dataset DEEPCRACK --epochs 200 --phase test --img_size 256 --model_name fang --batch_size 2 --optimizer_name Adam --print_interval 10 --save_interval 1  --lr 0.005 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset ER --epochs 200 --phase train --img_size 256 --model_name fang --batch_size 2 --optimizer_name Adam --print_interval 5 --save_interval 1  --lr 0.05 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score
# python main.py --dataset MITO --epochs 200 --phase train --img_size 256 --model_name fang --batch_size 2 --optimizer_name Adam --print_interval 5 --save_interval 1  --lr 0.05 --metric_list DSC ACC SEN SPE IoU PRE recall F1_score


## fang 的 chase 把 lr 设置为 0.005 不可行
## 但drive stare chase 把学习率设置成 0.008 更好了

## fang DEEPCRACK lr 0.008 dice 0.8315; lr 0.005 dice 0.8410 



# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name rollingunet --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet_resnet --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet_resnet --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet_resnet --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1


# python main.py --dataset CHASE_DB1 --epochs 2 --phase train --img_size 256 --model_name unet3_resnet --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet3_resnet --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet3_resnet --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1


# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet2_res_edge --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet2_res_edge --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet2_res_edge --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet2_res_ske --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet2_res_ske --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet2_res_ske --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_hidi --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_hidi --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_hidi --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_emi --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_emi --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_emi --batch_size 2 --optimizer_name Adam --lr 0.005 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet2_ske --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase train --img_size 256 --model_name unet2_ske --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase train --img_size 256 --model_name unet2_ske --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1


# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet2_edge --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase train --img_size 256 --model_name unet2_edge --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase train --img_size 256 --model_name unet2_edge --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1



# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name fang2 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 
# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_hidi1 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_hidi1 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase train --img_size 256 --model_name unet3_resnet_hidi1 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name fang2 --batch_size 2 --optimizer_name Adam --print_interval 1 --save_interval 1  --lr 0.0085 --metric_list DSC ACC SEN SPE IoU PRE 


# python main.py --dataset CHASE_DB1 --epochs 2 --phase train --img_size 256 --model_name unet3_skele_test --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet3_skele --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet3_skele --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet3_3ske --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet3_3ske --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet3_3ske --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1


# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet3_3ske_1019 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet3_3ske_1019 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet3_3ske_1019 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1


# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet3_3ske_1020_norm --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset DRIVE --epochs 200 --phase test --img_size 256 --model_name unet3_3ske_1020_norm --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1
# python main.py --dataset STARE --epochs 200 --phase test --img_size 256 --model_name unet3_3ske_1020_norm --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1

# python main.py --dataset CHASE_DB1 --epochs 300 --phase train --img_size 256 --model_name unet1220 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1


# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet1221 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --output_end chan
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet1221 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --output_end chan


# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --output_end lr0.0085
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --output_end lr0.0085

# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --optimizer_name SGD --output_end SGD
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --optimizer_name SGD --output_end SGD


# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.01 --print_interval 1 --save_interval 1  --output_end lr0.01_0.3mi
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.01 --print_interval 1 --save_interval 1  --output_end lr0.01_0.3mi
# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.0005 --print_interval 1 --save_interval 1  --output_end local_correlation0.0005
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet1231 --batch_size 2 --lr 0.0005 --print_interval 1 --save_interval 1  --output_end local_correlation0.0005

# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet1221 --batch_size 2 --lr 0.001 --print_interval 1 --save_interval 1 --output_end local_correlation_add1
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet1221 --batch_size 2 --lr 0.001 --print_interval 1 --save_interval 1 --output_end local_correlation_add1


# python main.py --dataset CHASE_DB1 --epochs 50 --phase train --img_size 256 --model_name unet0101 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --output_end loss
# python main.py --dataset CHASE_DB1 --epochs 50 --phase test --img_size 256 --model_name unet0101 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1 --output_end loss

# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 256 --model_name unet0107 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1  --output_end adddata1
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 256 --model_name unet0107 --batch_size 2 --lr 0.0085 --print_interval 1 --save_interval 1  --output_end adddata1

# python main.py --dataset CHASE_DB1 --epochs 300 --phase train --img_size 256 --model_name unet0112 --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1  --output_end ssim_0.84
# python main.py --dataset CHASE_DB1 --epochs 300 --phase test --img_size 256 --model_name unet0112 --batch_size 2 --lr 0.005 --print_interval 1 --save_interval 1  --output_end ssim_0.84

# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 224 --model_name swinunet --batch_size 2 --lr 0.01 --print_interval 1 --save_interval 1  --output_end 0.01-sgd --optimizer_name SGD
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 224 --model_name swinunet --batch_size 2 --lr 0.01 --print_interval 1 --save_interval 1  --output_end 0.01-sgd --optimizer_name SGD
# python main.py --dataset CHASE_DB1 --epochs 200 --phase train --img_size 224 --model_name swinunet --batch_size 2 --lr 0.01 --print_interval 1 --save_interval 1  --output_end 0.01-adam 
# python main.py --dataset CHASE_DB1 --epochs 200 --phase test --img_size 224 --model_name swinunet --batch_size 2 --lr 0.01 --print_interval 1 --save_interval 1  --output_end 0.01-adam 


# python main.py --dataset CHASE_DB1 --epochs 300 --phase train --img_size 512 --model_name EPSS --batch_size 2 --lr 0.0085 --output_1 lr0.0085 --print_interval 1 --save_interval 1 
# python main.py --dataset CHASE_DB1 --epochs 300 --phase test --img_size 512 --model_name EPSS --batch_size 2 --lr 0.0085 --output_end 1 --print_interval 1 --save_interval 1 
