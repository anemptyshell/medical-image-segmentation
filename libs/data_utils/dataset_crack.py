import os
import os.path
from torchvision import transforms
import numpy as np
import scipy.misc as m
import torch.utils.data as data
import cv2
from torch.utils import data


class Crackloader(data.Dataset):

    def __init__(self, txt_path, transform=None,normalize=True):
        self.txt_path = txt_path
        # self.root = "/home/nlg/CrackNet/datasets/DeepCrack-DS/train/"

        if normalize:
            self.img_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.img_transforms = transforms.ToTensor()

        self.train_set_path = self.make_dataset(txt_path)

    def __len__(self):
        return len(self.train_set_path)

    def __getitem__(self, index):
        img_path, lbl_path = self.train_set_path[index]
        # img = m.imread(self.root + img_path, mode='RGB')
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # img=cv2.resize(img,(400,400))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        img = np.array(img, dtype=np.uint8)

        img = self.img_transforms(img)

        lbl = cv2.imread(lbl_path)

        lbl = np.array(lbl, dtype=np.uint8)
        lbl = lbl[:, :, 1]

        _, binary = cv2.threshold(lbl,127, 1, cv2.THRESH_BINARY)
        # num_positive = np.sum((binary == 1))
        # num_negative = np.sum((binary == 0))
        return img, binary

    def make_dataset(self, txt_path):
        dataset = []
        index=0
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                # print(index,line)
                index+=1
                line = ''.join(line).strip()
                line_list = line.split(' ')
                dataset.append([line_list[0], line_list[1]])
        return dataset