import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt
import os
from skimage import morphology
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

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


# def skeleton_extract(root):
#     img_root = os.path.join(root, 'masks')
#     skeleton_root = os.path.join(root, 'masks_skeleton')
#     if not os.path.exists(skeleton_root):
#         os.mkdir(skeleton_root)

#     file_names = os.listdir(img_root)
#     for name in file_names:
#         img = cv2.imread(os.path.join(img_root, name), -1)
#         img[img <= 100] = 0
#         img[img > 100] = 1
#         skeleton0 = morphology.skeletonize(img)
#         skeleton = skeleton0.astype(np.uint8) * 255
#         cv2.imwrite(os.path.join(skeleton_root, name), skeleton)

#     return 0


def edge_extract(root):
    img_root = os.path.join(root, 'masks')
    edge_root = os.path.join(root, 'masks_edges1')

    if not os.path.exists(edge_root):
        os.makedirs(edge_root)  # 创建目录，支持多级目录
    file_names = os.listdir(img_root)

    for name in file_names:
        img_path = os.path.join(img_root, name)
        cap = cv2.VideoCapture(img_path)
        ret, img = cap.read()
        cap.release()
        
        if not ret:
            print(f"无法读取图像: {name}")
            continue
            
        # 转换为灰度图
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
        
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(contour_img, contours, -1, 255, 1)
        base_name = os.path.splitext(name)[0]  
        output_name = f"{base_name}.png" 
        print(contour_img)
        break
        
        # save_path = os.path.join(edge_root, output_name)
        # if cv2.imwrite(save_path, contour_img):
        #     print(f"成功保存: {save_path}")
        # else:
        #     print(f"保存失败: {save_path}")
        
    return 0


# def skeleton_extract(root):
#     img_root = os.path.join(root, 'masks')
#     skeleton_root = os.path.join(root, 'skeleton_1')
    
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



def generate_custom_skeleton_alternative(binary_image, a=1):
    """
    根据骨架点处的半径值来调整血管宽度
    - 半径 < a: 保留原血管 .
    - 半径 > a: 将血管宽度削减到半径a
    """
    
    edt = distance_transform_edt(binary_image)
    skeleton = skeletonize(binary_image)
    skeleton_radii = np.where(skeleton, edt, 0)
    
    # 创建自定义骨架区域
    custom_skeleton = np.zeros_like(binary_image, dtype=bool)
    
    # 获取所有骨架点坐标和半径
    y_coords, x_coords = np.where(skeleton)
    radii = skeleton_radii[skeleton]
    
    # 创建坐标网格
    y_grid, x_grid = np.indices(binary_image.shape)
    
    for y, x, radius in zip(y_coords, x_coords, radii):
        # 计算图像中每个像素到当前骨架点 (y, x) 的距离。
        dist_to_center = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
        
        if radius >= a:
            # 半径大于a：创建半径为a的圆形区域
            circle_region = dist_to_center <= a
            custom_skeleton = np.logical_or(custom_skeleton, circle_region)
    
    # 对于半径<=a的区域，我们直接使用原始血管图像
    # 但需要确保这些区域不会被过度削减
    mask_radius_leq_a = np.zeros_like(binary_image, dtype=bool)
    
    for y, x, radius in zip(y_coords, x_coords, radii):
        if radius <= a:
            dist_to_center = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
            circle_region = dist_to_center <= radius  # 使用原始半径
            mask_radius_leq_a = np.logical_or(mask_radius_leq_a, circle_region)
    
    # 合并结果：半径<=a的区域 + 半径>a的削减区域
    custom_skeleton = np.logical_or(custom_skeleton, mask_radius_leq_a) 
    # 确保不超过原始血管边界
    # custom_skeleton = np.logical_and(custom_skeleton, binary_image.astype(bool))
    
    return custom_skeleton




def skeleton_extract(root, a):
    img_root = os.path.join(root, 'masks')
    skeleton_root = os.path.join(root, f'strong_{a}')
    
    if not os.path.exists(skeleton_root):
        os.makedirs(skeleton_root)  

    file_names = os.listdir(img_root)
    for name in file_names:
        img_path = os.path.join(img_root, name)
        if name.lower().endswith('.gif'):
            cap = cv2.VideoCapture(img_path)
            ret, img = cap.read()
            cap.release()
            if not ret:
                print(f"无法读取GIF图像: {name}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(img_path, -1)
            if img is None:
                print(f"无法读取图像: {name}")
                continue
        img = img.astype(np.uint8)
        img[img <= 100] = 0
        img[img > 100] = 1
        custom_skeleton = generate_custom_skeleton_alternative(img, a=a)
        custom_skeleton_uint8 = custom_skeleton.astype(np.uint8) * 255

        base_name = os.path.splitext(name)[0]
        output_name = f"{base_name}.png"
        save_path = os.path.join(skeleton_root, output_name)

        if cv2.imwrite(save_path, custom_skeleton_uint8):
            print(f"成功保存骨架图像: {save_path}")
        else:
            print(f"保存骨架图像失败: {save_path}")
    
    return 0
 



if __name__ == '__main__':
    # train_er = "/home/my/data/CHASE_DB1/train/"
    # # edge_extract(train_er)
    # for a in [0.5, 1, 1.5]:
    #     skeleton_extract(train_er, a)

    # train_er1 = "/home/my/data/DRIVE/train/"
    # for a in [0.5, 1, 1.5]:
    #     skeleton_extract(train_er1, a)
    
    train_er2 = "/home/my/data/STARE/train/"
    for a in [1, 1.5]:
        skeleton_extract(train_er2, a)


    