import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
import argparse
import os
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='../data/MITO/train/images')
    # parser.add_argument('--wavelet_path', default='../wavelet/CHASE_DB1')
    parser.add_argument('--L_path', default='../wavelet/MITO/L')
    parser.add_argument('--H_path', default='../wavelet/MITO/H')
    parser.add_argument('--Con_path', default='../wavelet/MITO/Con')
    parser.add_argument('--wavelet_type', default='dmey', help='haar, db2, bior1.5, bior2.4, coif1, dmey')
    parser.add_argument('--LL_ratio', default=0.2, type=float)
    parser.add_argument('--if_RGB', default=False)
    args = parser.parse_args()


    if not os.path.exists(args.L_path):
        os.makedirs(args.L_path)
    if not os.path.exists(args.H_path):
        os.makedirs(args.H_path)
    if not os.path.exists(args.Con_path):
        os.makedirs(args.Con_path)

    for i in os.listdir(args.image_path):
        image_path = os.path.join(args.image_path, i)
        # wavelet_path = os.path.join(args.wavelet_path, i)
        L_path = os.path.join(args.L_path, i)
        H_path = os.path.join(args.H_path, i)
        Con_path = os.path.join(args.Con_path, i)


        # image = Image.open(image_path)
        img_u8 = cv2.imread(image_path)
        image = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # image = Image.open(image_path)
        # image = np.array(image)
        # print(image.shape)  ## (960, 999, 3)

        LL, (LH, HL, HH) = pywt.dwt2(image, args.wavelet_type)

        LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255 

        # LL = Image.fromarray(LL.astype(np.uint8))
        # LL.save(L_path)

        LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255 
        HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255 
        HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255 

        AH = np.concatenate([LL, LH], axis=1)
        VD = np.concatenate([HL, HH], axis=1)
        img = np.concatenate([AH, VD], axis=0)
     
        # plt.figure(figsize=(999,960),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        plt.imshow(img, cmap='gray') 
        plt.axis('off')  
        plt.savefig(Con_path,bbox_inches='tight', pad_inches = 0)
        plt.close()


        # plt.imshow(img, 'gray')
        # plt.title('img')
        # plt.show()

        # break

        merge1 = HH + HL + LH
        merge1 = (merge1-merge1.min()) / (merge1.max()-merge1.min()) * 255

        # plt.figure(figsize=(size,size),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        plt.imshow(merge1, cmap='gray') 
        plt.axis('off')  
        plt.savefig(H_path,bbox_inches='tight', pad_inches = 0)
        plt.close()

        # plt.imshow(merge1, 'gray')
        # plt.title('merge1')
        # plt.show()

        # merge1 = Image.fromarray(merge1.astype(np.uint8))
        # merge1.save(H_path)

        # merge2 = merge1 + args.LL_ratio * LL
        # merge2 = (merge2-merge2.min()) / (merge2.max()-merge2.min()) * 255

        # merge2 = Image.fromarray(merge2.astype(np.uint8))
        # merge2.save(wavelet_path)







