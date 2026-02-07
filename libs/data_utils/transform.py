import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import cv2



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



def get_transform(parser, opt):
    
    opt = parser.get_args()
    train_transformer = transforms.Compose([
        myNormalize(opt.dataset, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(opt.img_size, opt.img_size)
    ])
    test_transformer = transforms.Compose([
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
