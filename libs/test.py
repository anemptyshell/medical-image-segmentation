import numpy as np
from skimage.morphology import skeletonize, dilation, disk
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import cv2

def generate_custom_skeleton(binary_image, a=1):
    """
    生成自定义宽度的血管骨架图
    参数：
        binary_image: 二值化血管图像 (H, W)，前景（血管）为1，背景为0
        a: 半径阈值（像素）
    返回：
        custom_skeleton: 自定义宽度骨架图 (H, W)
    """
    # 步骤1: 计算欧氏距离变换 (EDT)
    # 注意：输入必须是整数二值图，前景1，背景0
    edt = distance_transform_edt(binary_image)

    skeleton = skeletonize(binary_image)
    
    # 步骤3: 创建半径调整图
    adjusted_radius = np.minimum(edt, a)  # 取 min(r, a)
    print(adjusted_radius)
    
    # 步骤4: 创建自适应结构元素
    max_radius = int(np.ceil(a))
    selem = disk(max_radius)  # 创建最大所需半径的圆形结构元素
    
    # 步骤5: 创建骨架点权重图
    # 仅在骨架点位置保留调整后的半径值，其他位置为0
    radius_map = np.where(skeleton, adjusted_radius, 0)
    
    # 步骤6: 生成自定义宽度骨架
    custom_skeleton = np.zeros_like(binary_image, dtype=bool)
    for r in range(1, max_radius + 1):
        # 获取当前半径对应的骨架点
        current_points = (radius_map >= r) & (radius_map > 0)
        
        # 使用当前半径进行膨胀
        current_disk = disk(r)
        dilated = dilation(current_points, footprint=current_disk)
        
        # 叠加到结果
        custom_skeleton |= dilated

    return edt, adjusted_radius, skeleton, custom_skeleton

# 使用示例 --------------------------------------------------
# 假设 binary_image 是您的二值化视网膜图像 (584, 565)
# 生成自定义骨架图 (a=3)
# custom_skel = generate_custom_skeleton(binary_image, a=3)

# # 可视化结果
# fig, ax = plt.subplots(1, 2, figsize=(15, 7))
# ax[0].imshow(binary_image, cmap='gray')
# ax[0].set_title('Original Vessels')
# ax[1].imshow(custom_skel, cmap='gray')
# ax[1].set_title(f'Custom Skeleton (a={3})')
# plt.show()

# 性能优化版本（内存占用更低）---------------------------------
def generate_custom_skeleton_efficient(binary_image, a=3):
    """内存优化版本，适合大图像"""
    from skimage.morphology import disk
    
    # 计算距离变换和骨架
    edt = distance_transform_edt(binary_image)
    skeleton = skeletonize(binary_image)
    
    # 创建结果容器
    custom_skeleton = np.zeros_like(binary_image, dtype=bool)
    max_radius = int(np.ceil(a))
    
    # 预生成结构元素字典
    disks = {r: disk(r) for r in range(1, max_radius+1)}
    
    # 处理每个骨架点
    y, x = np.where(skeleton)
    for i, j in zip(y, x):
        r_val = min(edt[i, j], a)
        radius = int(np.ceil(r_val))
        
        if radius > 0:
            # 计算有效圆形区域
            r_min = max(0, i - radius)
            r_max = min(binary_image.shape[0], i + radius + 1)
            c_min = max(0, j - radius)
            c_max = min(binary_image.shape[1], j + radius + 1)
            
            # 创建局部区域
            local_slice = (slice(r_min, r_max), slice(c_min, c_max))
            local_selem = disks[radius]
            
            # 中心对齐
            selem_center = radius
            center_i = min(i - r_min, selem_center)
            center_j = min(j - c_min, selem_center)
            
            # 裁剪结构元素以适配边界
            selem_slice = (
                slice(selem_center - center_i, selem_center + (r_max - r_min) - center_i),
                slice(selem_center - center_j, selem_center + (c_max - c_min) - center_j)
            )
            cropped_selem = local_selem[selem_slice[0], selem_slice[1]]
            
            # 应用膨胀
            custom_skeleton[local_slice] |= cropped_selem

    return custom_skeleton



def generate_custom_skeleton_2(binary_image, a=1):
    """
    生成自定义宽度的血管骨架图
    参数：
    binary_image: 二值化血管图像 (H, W)，前景（血管）为1，背景为0
    a: 半径阈值（像素）
    返回：
    edt: 欧氏距离变换图
    adjusted_radius: 调整后的半径图
    skeleton: 骨架图
    custom_skeleton: 自定义宽度骨架图 (H, W)
    """
    # 步骤1: 计算欧氏距离变换 (EDT)
    edt = distance_transform_edt(binary_image)
    
    # 步骤2: 提取骨架（中心线）
    skeleton = skeletonize(binary_image)
    
    # 步骤3: 获取骨架点处的半径值
    skeleton_radius = np.where(skeleton, edt, 0)
    
    # 步骤4: 应用阈值a，限制最大半径
    adjusted_radius = np.minimum(skeleton_radius, a)
    
    # 步骤5: 生成自定义宽度骨架
    custom_skeleton = np.zeros_like(binary_image, dtype=bool)
    max_radius = int(np.ceil(a))
    
    # 为每个可能的半径值创建膨胀
    for r in range(1, max_radius + 1):
        # 获取需要以半径r膨胀的骨架点
        current_points = (adjusted_radius >= r) & (skeleton)
        
        # 使用当前半径进行膨胀
        current_disk = disk(r)
        dilated = dilation(current_points, footprint=current_disk)
        
        # 叠加到结果
        custom_skeleton |= dilated
    
    return edt, adjusted_radius, skeleton, custom_skeleton







image_path = "21.png"

cap = cv2.VideoCapture(image_path)
ret, img = cap.read()
cap.release()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype(np.uint8)
img[img <= 100] = 0
img[img > 100] = 1


# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# binary_image = img.astype(np.uint8)
# _, binary_image = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
# skeleton, custom_skel = generate_custom_skeleton(img, a=3)

# 大图像使用优化版
# custom_skel = generate_custom_skeleton_efficient(binary_image, a=3)

# 调整阈值a观察效果
for a in [1,2,3]:
    edt, adjusted_radius, skeleton, custom_skeleton = generate_custom_skeleton_2(img, a=a)
    plt.imshow(custom_skeleton, cmap='gray')
    plt.title(f'Custom Skeleton (a={a})')
    plt.show()