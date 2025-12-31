import torchvision.transforms as transforms
# from PIL import Image
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import cv2
import albumentations as A


# myRandomAffine = transforms.RandomAffine(
#     degrees=0,
#     translate=(0.05, 0.05),  # 轻微平移
#     scale=(0.9, 1.1),  # 轻微缩放
#     shear=(-5, 5)  # 轻微剪切
# )

# myElasticTransform = ElasticTransform(
#     alpha=50,  # 弹性变形强度
#     sigma=5  # 高斯滤波参数
# )

myElasticTransform = A.ElasticTransform(
        alpha=50,
        sigma=5,
        alpha_affine=0,
        p=0.5
    )


class myColorJitter(object):
    def __init__(self, p=0.5):
        self.p = p
        # 针对视网膜图像，调整范围较小
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
    
    def __call__(self, sample):
        if random.random() < self.p:
            image, mask = sample['image'], sample['mask']
            
            # 只对图像进行颜色增强，掩码保持不变
            transform = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast
            )
            image = transform(image)
            
            return {'image': image, 'mask': mask}
        return sample


class myVesselSpecificAug(object):
    """针对血管特征的增强"""
    def __init__(self, p=0.3):
        self.p = p
        
    def __call__(self, sample):
        if random.random() < self.p:
            image, mask = sample['image'], sample['mask']
            
            # 方法1：添加血管状噪声
            if random.random() < 0.5:
                image = self.add_vessel_like_noise(image, mask)
            
            # 方法2：模拟不同对比度的血管
            image = self.adjust_vessel_contrast(image, mask)
            
            return {'image': image, 'mask': mask}
        return sample
    
    def add_vessel_like_noise(self, image, mask):
        # 在血管附近添加细长噪声
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
        dilated_mask = cv2.dilate(mask.numpy(), kernel, iterations=1)
        noise = np.random.normal(0, 0.02, image.shape)
        noise_mask = (dilated_mask - mask.numpy()) > 0
        image[noise_mask] += noise[noise_mask] * 0.3
        return torch.clamp(image, 0, 1)


class myMixUp(object):
    """混合两张图像（谨慎使用）"""
    def __init__(self, p=0.3, alpha=0.2):
        self.p = p
        self.alpha = alpha
        
    def __call__(self, sample):
        if random.random() < self.p and hasattr(self, 'other_sample'):
            lam = np.random.beta(self.alpha, self.alpha)
            image1, mask1 = sample['image'], sample['mask']
            image2, mask2 = self.other_sample['image'], self.other_sample['mask']
            
            mixed_image = lam * image1 + (1 - lam) * image2
            mixed_mask = torch.logical_or(mask1 > 0.5, mask2 > 0.5).float()
            
            return {'image': mixed_image, 'mask': mixed_mask}
        return sample



"""isic17/18"""
class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, image_size_h=256, image_size_w=256):
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.image_size_h, self.image_size_w], antialias=True), TF.resize(mask, [self.image_size_h, self.image_size_w], antialias=True)
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 


class myNormalize:
    def __init__(self, dataset, train=True):
        if dataset == 'isic2018':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif dataset == 'isic2017':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        # elif dataset == 'isic18_82':
        else:
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk




def get_transform_v2(parser, opt):
    opt = parser.get_args()
    
    # 基础增强
    base_transforms = [
        myNormalize(opt.dataset, train=True),
        myToTensor(),
    ]
    
    # 几何变换增强
    geometric_transforms = [
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        # myRandomAffine(translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5, p=0.3),
        transforms.RandomAffine(  # 使用 transforms. 前缀
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=(-5, 5)
        ),
        # myElasticTransform(alpha=30, sigma=5, p=0.2),
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            alpha_affine=0,
            p=0.5
        ),
    ]
    
    # 外观变换增强（针对视网膜图像调整参数）
    appearance_transforms = [
        myColorJitter(brightness=0.1, contrast=0.1, p=0.3),  # 轻微调整
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),  # 轻微模糊
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    ]
    
    # 血管特异性增强
    vessel_specific = [
        myVesselSpecificAug(p=0.3),
    ]
    
    # 组合增强（注意顺序）
    train_transformer = transforms.Compose([
        *base_transforms,
        *geometric_transforms,
        *appearance_transforms,
        *vessel_specific,
        myResize(opt.img_size, opt.img_size),
        
        # 可选的增强
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])
    
    # 测试时仅使用基本变换
    test_transformer = transforms.Compose([
        myNormalize(opt.dataset, train=False),
        myToTensor(),
        myResize(opt.img_size, opt.img_size)
    ])
    
    return train_transformer, test_transformer