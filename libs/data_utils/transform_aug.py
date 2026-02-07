import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import cv2



"""isic17/18"""
# class myToTensor:
#     def __init__(self):
#         pass
#     def __call__(self, data):
#         image, mask = data
#         return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

# class myResize:
#     def __init__(self, image_size_h=256, image_size_w=256):
#         self.image_size_h = image_size_h
#         self.image_size_w = image_size_w
#     def __call__(self, data):
#         image, mask = data
#         return TF.resize(image, [self.image_size_h, self.image_size_w], antialias=True), TF.resize(mask, [self.image_size_h, self.image_size_w], antialias=True)
       

# class myRandomHorizontalFlip:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
#         else: return image, mask
            

# class myRandomVerticalFlip:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
#         else: return image, mask


# class myRandomRotation:
#     def __init__(self, p=0.5, degree=[0,360]):
#         self.angle = random.uniform(degree[0], degree[1])
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
#         else: return image, mask 


# class myNormalize:
#     def __init__(self, dataset, train=True):
#         if dataset == 'isic2018':
#             if train:
#                 self.mean = 157.561
#                 self.std = 26.706
#             else:
#                 self.mean = 149.034
#                 self.std = 32.022
#         elif dataset == 'isic2017':
#             if train:
#                 self.mean = 159.922
#                 self.std = 28.871
#             else:
#                 self.mean = 148.429
#                 self.std = 25.748
#         # elif dataset == 'isic18_82':
#         else:
#             if train:
#                 self.mean = 156.2899
#                 self.std = 26.5457
#             else:
#                 self.mean = 149.8485
#                 self.std = 35.3346
            
#     def __call__(self, data):
#         img, msk = data
#         img_normalized = (img-self.mean)/self.std
#         img_normalized = ((img_normalized - np.min(img_normalized)) 
#                             / (np.max(img_normalized)-np.min(img_normalized))) * 255.
#         return img_normalized, msk

 

# class convert_to_grayscale:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if len(image.shape) == 3:
#             # 使用加权平均法进行灰度化（OpenCV默认方法）
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             image = image.copy()
#         return image, mask

# class normalize_image:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         image_float = image.astype(np.float32)
#         # 获取最小值和最大值
#         min_val = np.min(image_float)
#         max_val = np.max(image_float)

#         # 避免除以零
#         if max_val - min_val == 0:
#             return image_float.astype(np.uint8)

#         # 归一化到0-255范围
#         normalized = ((image_float - min_val) / (max_val - min_val)) * 255

#         return normalized.astype(np.uint8), mask


# class apply_clahe:
#     def __init__(self, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
#         self.clahe_clip_limit = clahe_clip_limit
#         self.clahe_grid_size = clahe_grid_size
#     def __call__(self, data):
#         image, mask = data
#         # 创建CLAHE对象
#         clahe = cv2.createCLAHE(
#             clipLimit=self.clahe_clip_limit,
#             tileGridSize=self.clahe_grid_size
#         )

#         # 应用CLAHE
#         clahe_image = clahe.apply(image)

#         return clahe_image, mask
    
# class apply_gamma_correction:
#     def __init__(self, gamma=0.5):
#         self.gamma = gamma
#     def __call__(self, data):
#         image, mask = data
#         image_normalized = image.astype(np.float32) / 255.0
#         gamma_corrected = np.power(image_normalized, self.gamma)
#         gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

#         return gamma_corrected, mask

    


# def get_transform(parser, opt):
    
#     opt = parser.get_args()
#     train_transformer = transforms.Compose([
#         myNormalize(opt.dataset, train=True),
#         convert_to_grayscale(p=0.5),
#         apply_clahe(clahe_clip_limit=2.0, clahe_grid_size=(8, 8)),
#         apply_gamma_correction(gamma=0.5),
#         myToTensor(),
#         myRandomHorizontalFlip(p=0.5),
#         myRandomVerticalFlip(p=0.5),
#         myRandomRotation(p=0.5, degree=[0, 360]),
#         myResize(opt.img_size, opt.img_size)
#     ])
#     test_transformer = transforms.Compose([
#         myNormalize(opt.dataset, train=False),
#         convert_to_grayscale(p=0.5),
#         apply_clahe(clahe_clip_limit=2.0, clahe_grid_size=(8, 8)),
#         apply_gamma_correction(gamma=0.5),
#         myToTensor(),
#         myResize(opt.img_size, opt.img_size)
#     ])


#     return train_transformer, test_transformer


import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import random


class myToTensor:
    def __init__(self):
        pass
    
    def __call__(self, data):
        image, mask = data
        
        # 处理图像
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # 如果是灰度图，添加通道维度 [H, W] -> [1, H, W]
                image = np.expand_dims(image, axis=0)
            elif len(image.shape) == 3 and image.shape[2] in [1, 3]:  # [H, W, C] -> [C, H, W]
                image = image.transpose(2, 0, 1)
            # 转换为torch tensor
            image = torch.from_numpy(image.copy()).float()
        else:
            image = torch.tensor(image).float()
            if len(image.shape) == 2:  # 如果是灰度图，添加通道维度
                image = image.unsqueeze(0)
            elif len(image.shape) == 3 and image.shape[2] in [1, 3]:  # [H, W, C] -> [C, H, W]
                image = image.permute(2, 0, 1)
        
        # 处理mask
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 2:  # 如果是单通道mask
                mask = np.expand_dims(mask, axis=0)
            elif len(mask.shape) == 3 and mask.shape[2] == 1:  # [H, W, 1] -> [1, H, W]
                mask = mask.transpose(2, 0, 1)
            # 转换为torch tensor
            mask = torch.from_numpy(mask.copy()).float()
        else:
            mask = torch.tensor(mask).float()
            if len(mask.shape) == 2:  # 如果是单通道mask
                mask = mask.unsqueeze(0)
            elif len(mask.shape) == 3 and mask.shape[2] == 1:  # [H, W, 1] -> [1, H, W]
                mask = mask.permute(2, 0, 1)
        
        return image, mask


class myResize:
    def __init__(self, image_size_h=256, image_size_w=256):
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
    
    def __call__(self, data):
        image, mask = data
        
        # 确保是torch tensor
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).float()
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).float()
        
        # 调整维度顺序为 [C, H, W]
        if len(image.shape) == 3:
            if image.shape[0] > 3:  # 可能是 [H, W, C] 格式
                image = image.permute(2, 0, 1)
        else:
            image = image.unsqueeze(0)  # 添加通道维度
        
        if len(mask.shape) == 3:
            if mask.shape[0] > 1:  # 可能是 [H, W, C] 格式
                mask = mask.permute(2, 0, 1)
        else:
            mask = mask.unsqueeze(0)  # 添加通道维度
        
        # 调整大小
        image_resized = TF.resize(image, [self.image_size_h, self.image_size_w], antialias=True)
        mask_resized = TF.resize(mask, [self.image_size_h, self.image_size_w], antialias=True)
        
        return image_resized, mask_resized


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            if torch.is_tensor(image):
                return TF.hflip(image), TF.hflip(mask)
            else:
                return np.fliplr(image), np.fliplr(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            if torch.is_tensor(image):
                return TF.vflip(image), TF.vflip(mask)
            else:
                return np.flipud(image), np.flipud(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.p = p
        self.degree = degree
    
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            if torch.is_tensor(image):
                return TF.rotate(image, angle), TF.rotate(mask, angle)
            else:
                # 使用OpenCV旋转
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                mask_rotated = cv2.warpAffine(mask, M, (w, h))
                return image_rotated, mask_rotated
        else:
            return image, mask


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
        else:
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
    
    def __call__(self, data):
        img, msk = data
        
        # 如果是tensor，转换为numpy
        if torch.is_tensor(img):
            img = img.numpy()
        
        # 确保是浮点数
        img_float = img.astype(np.float32)
        
        # 应用归一化
        img_normalized = (img_float - self.mean) / self.std
        
        # 归一化到0-255范围
        min_val = np.min(img_normalized)
        max_val = np.max(img_normalized)
        
        if max_val - min_val > 0:
            img_normalized = ((img_normalized - min_val) / (max_val - min_val)) * 255.0
        else:
            img_normalized = img_normalized * 255.0
        
        return img_normalized.astype(np.uint8), msk


class convert_to_grayscale:
    def __init__(self, p=1.0):
        self.p = p
    
    def __call__(self, data):
        image, mask = data
        
        # 确保是numpy数组
        if torch.is_tensor(image):
            image = image.numpy()
        
        # 随机决定是否应用
        if random.random() < self.p:
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB转灰度
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # 已经是单通道，压缩维度
                image = image.squeeze(2)
        
        return image, mask


class apply_clahe:
    def __init__(self, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
    
    def __call__(self, data):
        image, mask = data
        
        # 确保是numpy数组
        if torch.is_tensor(image):
            image = image.numpy()
        
        # 确保是2D图像
        if len(image.shape) == 3:
            if image.shape[2] == 1 or image.shape[2] == 3:
                image = image.squeeze()
        
        # 确保是uint8类型
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 应用CLAHE
        try:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )
            clahe_image = clahe.apply(image)
        except Exception as e:
            print(f"Warning: CLAHE failed: {e}")
            clahe_image = image
        
        return clahe_image, mask


class apply_gamma_correction:
    def __init__(self, gamma=0.5):
        self.gamma = gamma
    
    def __call__(self, data):
        image, mask = data
        
        # 确保是numpy数组
        if torch.is_tensor(image):
            image = image.numpy()
        
        # 确保是uint8类型
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 应用gamma校正
        image_normalized = image.astype(np.float32) / 255.0
        gamma_corrected = np.power(image_normalized, self.gamma)
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        
        return gamma_corrected, mask


def get_transform(parser, opt):
    opt = parser.get_args()
    
    # 训练时的变换顺序
    train_transformer = transforms.Compose([
        convert_to_grayscale(p=1.0),
        apply_clahe(clahe_clip_limit=2.0, clahe_grid_size=(8, 8)),
        apply_gamma_correction(gamma=0.5),
        myNormalize(opt.dataset, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(opt.img_size, opt.img_size)
    ])
    
    # 测试时的变换顺序
    test_transformer = transforms.Compose([
        convert_to_grayscale(p=1.0),
        apply_clahe(clahe_clip_limit=2.0, clahe_grid_size=(8, 8)),
        apply_gamma_correction(gamma=0.5),
        myNormalize(opt.dataset, train=False),
        myToTensor(),
        myResize(opt.img_size, opt.img_size)
    ])
    
    return train_transformer, test_transformer




"""---------------------下面不用看---------------------------------"""


"""dconnnet 视网膜"""
def default_DRIVE_loader(parser, img_path, mask_path, train=False):
    # print(mask_path)
    # print("////////")
    opt = parser.get_args()
    img = cv2.imread(img_path)
    img = cv2.resize(img, (opt.img_size, opt.img_size))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask = np.array(Image.open(mask_path).convert('L'))    ### for DRIVE
    # mask = cv2.imread(mask_path)
    # print('mask',mask)
    # mask = np.array(Image.open(mask_path))
    # print(img.shape,mask.shape)
    mask = cv2.resize(mask, (opt.img_size, opt.img_size))
    if train:
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask

def default_DRIVE_loader_01(parser, thin_path, thick_path, train=False):

    opt = parser.get_args()
    thin = cv2.imread(thin_path, cv2.IMREAD_GRAYSCALE)
    thin = cv2.resize(thin, (opt.img_size, opt.img_size))
    thick = cv2.imread(thick_path, cv2.IMREAD_GRAYSCALE)
    thick = cv2.resize(thick, (opt.img_size, opt.img_size))
    if train:
        thin, thick = randomShiftScaleRotate(thin, thick,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        thin, thick = randomHorizontalFlip(thin, thick)
        thin, thick = randomVerticleFlip(thin, thick)
        thin, thick = randomRotate90(thin, thick)

    thin = np.expand_dims(thin, axis=2)
    thick = np.expand_dims(thick, axis=2)
    thin = np.array(thin, np.float32).transpose(2, 0, 1) / 255.0 
    thick = np.array(thick, np.float32).transpose(2, 0, 1) / 255.0

    thick[thick >= 0.5] = 1
    thick[thick <= 0.5] = 0
    thin[thin >= 0.5] = 1
    thin[thin <= 0.5] = 0

    return thin, thick


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

class Normalize(object):
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std

        image = (image-image.min())/(image.max()-image.min())
        mask = mask/255.0
        if mask is None:
            return image
        return image, mask

class RandomCrop(object):
    def __call__(self, image, mask=None):
        H,W   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1].copy(), mask[:,::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)

        return image, mask
    

"""ER"""
def img_PreProc_er(img, pro_type):
    if pro_type == "clahe":
        img = img_clahe(img)
        img = img / 65535.
        sd_img = standardization(img)
        return sd_img

    elif pro_type == "invert":
        img = 65535 - img
        return img / 65535.

    elif pro_type == "edgeEnhance":
        edge = sober_filter(img)
        edge = edge / np.max(edge)
        return ((img / 65535.) + edge) * 0.5

    elif pro_type == "norm":
        img = img / 65535.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img

    elif pro_type == "clahe_norm":
        img = img_clahe(img)
        img = img / 65535.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img

def img_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

def img_clahe_cm(img):
    b,g,r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    output = cv2.merge((b,g,r))
    return output

def img_normalized(img):
    std = np.std(img)
    mean = np.mean(img)
    img_normalized = (img - mean) / (std + 1e-10)
    return img_normalized


def convert_16to8(img):
    img = (img - np.mean(img)) / np.std(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    return img

def convert_8to16(img):
    img = (img - np.mean(img)) / np.std(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 65535).astype(np.uint16)
    return img

def sober_filter(img):
    if img.dtype == "uint16":
        dx = np.array(cv2.Sobel(img, cv2.CV_32F, 1, 0))
        dy = np.array(cv2.Sobel(img, cv2.CV_32F, 0, 1))
    elif img.dtype == "uint8":
        dx = np.array(cv2.Sobel(img, cv2.CV_16S, 1, 0))
        dy = np.array(cv2.Sobel(img, cv2.CV_16S, 0, 1))
    dx = np.abs(dx)
    dy = np.abs(dy)
    edge = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    return edge


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

def npy_PreProc(npy):
    img_FD = npy[0]
    img_FL = npy[1]
    FD_min = np.min(img_FD)
    FD_max = np.max(img_FD)
    img_FD = (img_FD - FD_min) / (FD_max - FD_min)

    FL_min = np.min(img_FL)
    FL_max = np.max(img_FL)
    img_FL = (img_FL - FL_min) / (FL_max - FL_min)
    sd_FD = standardization(img_FD)
    sd_FL = standardization(img_FL)
    return sd_FD, sd_FL
