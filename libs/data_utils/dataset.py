from PIL import Image
import os
import os.path
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import Dataset
from libs.data_utils.transform import get_transform, default_DRIVE_loader,img_PreProc_er, default_DRIVE_loader_01
# from libs.data_utils.transform_v2 import get_transform_v2
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import torch
import random
from scipy.ndimage.morphology import distance_transform_edt



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)




class ISIC_datasets(Dataset):
    def __init__(
        self,
        parser,
        args,
        # root,
        train=True
    ):
        super(ISIC_datasets, self)


        self.parser = parser
        self.args = self.parser.get_args()
        self.train = train
        # self.root=root
        
        self.train_transformer, self.test_transformer = get_transform(self.parser, self.args)

        if train:
            img_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "train/images/"))
            mask_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "train/masks/"))
            # thin_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "thin/"))
            # thick_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "thick/"))
            edge_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "skeleton_2/"))
            strong_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "strong_2/"))
            skeleton_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "skeleton_2/"))

            img_train_list = sorted(img_train_list)
            mask_train_list = sorted(mask_train_list)
            # thin_train_list = sorted(thin_train_list)
            # thick_train_list = sorted(thick_train_list)
            edge_train_list = sorted(edge_train_list)
            strong_train_list = sorted(strong_train_list)
            skeleton_train_list = sorted(skeleton_train_list)

            self.data_list = []
            for i in range(len(img_train_list)):
               img_train_path = os.path.join(args.data_path, args.dataset, "train/images/") + img_train_list[i]
               mask_train_path = os.path.join(args.data_path, args.dataset, "train/masks/") + mask_train_list[i]
            #    thin_train_path = os.path.join(args.data_path, args.dataset, "thin/") + thin_train_list[i]
            #    thick_train_path = os.path.join(args.data_path, args.dataset, "thick/") + thick_train_list[i]
               edge_train_path = os.path.join(args.data_path, args.dataset, "skeleton_2/") + edge_train_list[i]
               strong_train_path = os.path.join(args.data_path, args.dataset, "strong_2/") + strong_train_list[i]
               skeleton_train_path = os.path.join(args.data_path, args.dataset, "skeleton_2/") + skeleton_train_list[i]

            #    self.data_list.append([img_train_path, mask_train_path])   ## 列表
            #    self.data_list.append([img_train_path, mask_train_path, thin_train_path, thick_train_path])   ## 列表
               self.data_list.append([img_train_path, mask_train_path, strong_train_path, skeleton_train_path, edge_train_path])   ## 列表

            self.transformer = self.train_transformer

        else:
            img_test_list = os.listdir(os.path.join(args.data_path, args.dataset, "val/images/"))
            mask_test_list = os.listdir(os.path.join(args.data_path, args.dataset, "val/masks/"))
            img_test_list = sorted(img_test_list)
            mask_test_list = sorted(mask_test_list)

            self.data_list = []
            for i in range(len(img_test_list)):
               img_test_path = os.path.join(args.data_path, args.dataset, "val/images/") + img_test_list[i]
               mask_test_path = os.path.join(args.data_path, args.dataset, "val/masks/") + mask_test_list[i]
               self.data_list.append([img_test_path, mask_test_path])  
            self.transformer = self.test_transformer


    def __getitem__(self, index):

        if self.train:
            # img_path, mask_path, thin_path, thick_path = self.data_list[index]

            # thin = cv2.imread(thin_path, cv2.IMREAD_GRAYSCALE)
            # thin = cv2.resize(thin, (self.args.img_size, self.args.img_size))
            # thick = cv2.imread(thick_path, cv2.IMREAD_GRAYSCALE)
            # thick = cv2.resize(thick, (self.args.img_size, self.args.img_size))
            
            # thin_ = torch.from_numpy(thin / 255.).unsqueeze_(dim=0).float()
            # thick_ = torch.from_numpy(thick / 255.).unsqueeze_(dim=0).float()

            # thin = torch.Tensor(thin_)
            # thick = torch.Tensor(thick_)

            ## --------------------------------------------------------------------------

            img_path, mask_path, strong_path, skeleton_path, edge_path = self.data_list[index]
            
            strong = cv2.imread(strong_path, cv2.IMREAD_GRAYSCALE)
            strong = cv2.resize(strong, (self.args.img_size, self.args.img_size))
            skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
            skeleton = cv2.resize(skeleton, (self.args.img_size, self.args.img_size))
            edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            edge = cv2.resize(edge, (self.args.img_size, self.args.img_size))
            
            strong_ = torch.from_numpy(strong / 255.).unsqueeze_(dim=0).float()
            skeleton_ = torch.from_numpy(skeleton / 255.).unsqueeze_(dim=0).float()
            edge_ = torch.from_numpy(edge / 255.).unsqueeze_(dim=0).float()

            strong = torch.Tensor(strong_)
            skeleton = torch.Tensor(skeleton_)
            edge = torch.Tensor(edge_)
        else:
            img_path, mask_path = self.data_list[index]

        # img_path, mask_path = self.data_list[index]

        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255
        img, mask = self.transformer((img, mask))

        if self.train:
            # return img, mask, thin, thick
            return img, mask, strong, skeleton, edge
        else:
            return img, mask
        
        # return img, mask

    def __len__(self):
        return len(self.data_list)



class CHASE_datasets(data.Dataset):
    def __init__(self, parser, args, train=True): 
        super(CHASE_datasets, self)

        self.parser = parser
        self.args = self.parser.get_args()
        self.train = train

        if self.train:
            img_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "train/images/"))
            mask_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "train/masks/"))
            thin_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "thin/"))
            thick_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "thick/"))

            img_train_list = sorted(img_train_list)
            mask_train_list = sorted(mask_train_list)
            thin_train_list = sorted(thin_train_list)
            thick_train_list = sorted(thick_train_list)

            self.data_list = []
            for i in range(len(img_train_list)):
               img_train_path = os.path.join(args.data_path, args.dataset, "train/images/") + img_train_list[i]
               mask_train_path = os.path.join(args.data_path, args.dataset, "train/masks/") + mask_train_list[i]
               thin_train_path = os.path.join(args.data_path, args.dataset, "thin/") + thin_train_list[i]
               thick_train_path = os.path.join(args.data_path, args.dataset, "thick/") + thick_train_list[i]
               self.data_list.append([img_train_path, mask_train_path, thin_train_path, thick_train_path])   ## 列表

        else:
            img_test_list = os.listdir(os.path.join(args.data_path, args.dataset, "val/images/"))
            mask_test_list = os.listdir(os.path.join(args.data_path, args.dataset, "val/masks/"))
            img_test_list = sorted(img_test_list)
            mask_test_list = sorted(mask_test_list)

            self.data_list = []
            for i in range(len(img_test_list)):
               img_test_path = os.path.join(args.data_path, args.dataset, "val/images/") + img_test_list[i]
               mask_test_path = os.path.join(args.data_path, args.dataset, "val/masks/") + mask_test_list[i]
               self.data_list.append([img_test_path, mask_test_path])  


    def __getitem__(self, index):
        if self.train:
            img_path, mask_path, thin_path, thick_path = self.data_list[index]
            # thin, thick = default_DRIVE_loader_01(self.parser, thin_path, thick_path, self.train)
            thin = cv2.imread(thin_path, cv2.IMREAD_GRAYSCALE)
            thin = cv2.resize(thin, (self.args.img_size, self.args.img_size))
            thick = cv2.imread(thick_path, cv2.IMREAD_GRAYSCALE)
            thick = cv2.resize(thick, (self.args.img_size, self.args.img_size))

            thin_ = torch.from_numpy(thin / 255.).unsqueeze_(dim=0).float()
            thick_ = torch.from_numpy(thick / 255.).unsqueeze_(dim=0).float()

            thin = torch.Tensor(thin_)
            thick = torch.Tensor(thick_)
        else:
            img_path, mask_path = self.data_list[index]

        img, mask = default_DRIVE_loader(self.parser, img_path, mask_path, self.train)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        if self.train:
            return img.squeeze(0), mask, thin, thick
        else:
            return img.squeeze(0), mask

    def __len__(self):
        return len(self.data_list)


        
class ER_datasets(data.Dataset):
    def __init__(self, parser, args, train=True): 
        super(ER_datasets, self)

        self.parser = parser
        self.args = self.parser.get_args()
        self.train = train
        # self.train_transformer, self.test_transformer = get_transform(self.parser, self.args)

        if self.train:
            img_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "train/images/"))
            mask_train_list = os.listdir(os.path.join(args.data_path, args.dataset, "train/masks/"))
            img_train_list = sorted(img_train_list)
            mask_train_list = sorted(mask_train_list)

            self.data_list = []
            for i in range(len(img_train_list)):
               img_train_path = os.path.join(args.data_path, args.dataset, "train/images/") + img_train_list[i]
               mask_train_path = os.path.join(args.data_path, args.dataset, "train/masks/") + mask_train_list[i]
               self.data_list.append([img_train_path, mask_train_path])   ## 列表
            # self.transformer = self.train_transformer

        else:
            img_test_list = os.listdir(os.path.join(args.data_path, args.dataset, "val/images/"))
            mask_test_list = os.listdir(os.path.join(args.data_path, args.dataset, "val/masks/"))
            img_test_list = sorted(img_test_list)
            mask_test_list = sorted(mask_test_list)

            self.data_list = []
            for i in range(len(img_test_list)):
               img_test_path = os.path.join(args.data_path, args.dataset, "val/images/") + img_test_list[i]
               mask_test_path = os.path.join(args.data_path, args.dataset, "val/masks/") + mask_test_list[i]
               self.data_list.append([img_test_path, mask_test_path])  
            # self.transformer = self.test_transformer

    def __getitem__(self, index):
        img_path, mask_path = self.data_list[index]
        # print(img_path)
        # print(mask_path)
        # print('-----------------')

        # img = cv2.imread(img_path, -1)
        # mask = cv2.imread(mask_path, -1)
        # img_ = img_PreProc_er(img, pro_type='clahe')
        # img_ = torch.from_numpy(img_).unsqueeze(dim=0).float()
        # # print(img_.size())    ## torch.Size([1, 256, 256])
        # mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

        # img = np.array(Image.open(img_path).convert('RGB'))
        # mask = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255

        img = cv2.imread(img_path, -1)
        mask = cv2.imread(mask_path, -1)

        img_ = img_PreProc_er(img, pro_type='clahe')
        img_ = torch.from_numpy(img_).unsqueeze(dim=0).float()
        mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()



        # img, mask = self.transformer((img, mask))

        return img_, mask_
        # return img, mask


    def __len__(self):
        return len(self.data_list)


class crack_datasets(data.Dataset):
   def __init__(self, parser, args, train=True): 
        super(crack_datasets, self)

        self.parser = parser
        self.args = self.parser.get_args()
        self.train = train





















