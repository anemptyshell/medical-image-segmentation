import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt
import os
from skimage import morphology


def edge_extract(root):
    img_root = os.path.join(root, 'masks')
    edge_root = os.path.join(root, 'masks_edges1')

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)
    index = 0
    for name in file_names:
        img = cv2.imread(os.path.join(img_root, name), 0)
        edge, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(img)
        cv2.drawContours(contour_img, edge, -1, (255), 1)
        print(contour_img)
        cv2.imwrite(os.path.join(edge_root, name), contour_img)
        index += 1
    return 0


def skeleton_extract(root):
    img_root = os.path.join(root, 'masks')
    skeleton_root = os.path.join(root, 'masks_skeleton')
    if not os.path.exists(skeleton_root):
        os.mkdir(skeleton_root)

    file_names = os.listdir(img_root)
    for name in file_names:
        img = cv2.imread(os.path.join(img_root, name), -1)
        img[img <= 100] = 0
        img[img > 100] = 1
        skeleton0 = morphology.skeletonize(img)
        skeleton = skeleton0.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(skeleton_root, name), skeleton)

    return 0


# def edge_extract(root):
#     img_root = os.path.join(root, 'masks')
#     edge_root = os.path.join(root, 'masks_edges1')

#     if not os.path.exists(edge_root):
#         os.makedirs(edge_root)  # 创建目录，支持多级目录
#     file_names = os.listdir(img_root)

#     for name in file_names:
#         img_path = os.path.join(img_root, name)
#         cap = cv2.VideoCapture(img_path)
#         ret, img = cap.read()
#         cap.release()
        
#         if not ret:
#             print(f"无法读取图像: {name}")
#             continue
            
#         # 转换为灰度图
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = img.astype(np.uint8)
        
#         contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contour_img = np.zeros_like(img, dtype=np.uint8)
#         cv2.drawContours(contour_img, contours, -1, 255, 1)
#         base_name = os.path.splitext(name)[0]  
#         output_name = f"{base_name}.png" 
#         print(contour_img)
#         break
        
#         # save_path = os.path.join(edge_root, output_name)
#         # if cv2.imwrite(save_path, contour_img):
#         #     print(f"成功保存: {save_path}")
#         # else:
#         #     print(f"保存失败: {save_path}")
        
#     return 0


# def skeleton_extract(root):
#     img_root = os.path.join(root, 'masks')
#     skeleton_root = os.path.join(root, 'masks_skeleton')
    
#     if not os.path.exists(skeleton_root):
#         os.makedirs(skeleton_root)  

#     file_names = os.listdir(img_root)
#     for name in file_names:
#         img_path = os.path.join(img_root, name)
        
#         if name.lower().endswith('.gif'):
#             cap = cv2.VideoCapture(img_path)
#             ret, img = cap.read()
#             cap.release()
#             if not ret:
#                 print(f"无法读取GIF图像: {name}")
#                 continue
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         else:
#             img = cv2.imread(img_path, -1)
#             if img is None:
#                 print(f"无法读取图像: {name}")
#                 continue

#         img = img.astype(np.uint8)
#         img[img <= 100] = 0
#         img[img > 100] = 1
#         skeleton0 = morphology.skeletonize(img)
#         skeleton = skeleton0.astype(np.uint8) * 255
#         base_name = os.path.splitext(name)[0]
#         output_name = f"{base_name}.png"
#         save_path = os.path.join(skeleton_root, output_name)
#         print(skeleton.shape)
        
#         # if cv2.imwrite(save_path, skeleton):
#         #     print(f"成功保存骨架图像: {save_path}")
#         # else:
#         #     print(f"保存骨架图像失败: {save_path}")

#     return 0

if __name__ == '__main__':
    train_er = "/home/my/data/DRIVE/train/"
    # edge_extract(train_er)
    skeleton_extract(train_er)
    