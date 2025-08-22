import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import cv2

def generate_custom_skeleton(binary_image, a=1):
    """
    生成自定义宽度的血管骨架图
    参数：
    binary_image: 二值化血管图像 (H, W)，前景（血管）为1，背景为0
    a: 距离阈值（像素）
    返回：
    edt: 欧氏距离变换图
    skeleton: 骨架图
    custom_skeleton: 自定义宽度骨架图 (H, W)
    """
    # 步骤1: 计算欧氏距离变换 (EDT)
    edt = distance_transform_edt(binary_image)
    
    # 步骤2: 提取骨架（中心线）
    skeleton = skeletonize(binary_image)
    
    # 步骤3: 获取骨架点处的距离值
    skeleton_distances = np.where(skeleton, edt, 0)
    
    # 步骤4: 应用阈值a，限制最大距离
    adjusted_distances = np.minimum(skeleton_distances, a)
    
    # 步骤5: 创建距离掩码 - 这是关键步骤
    # 对于每个像素，计算它到最近骨架点的距离
    # 如果这个距离 <= 该骨架点对应的调整后距离，则保留
    
    # 首先创建骨架点的坐标和对应距离的映射
    skeleton_points = np.where(skeleton)
    skeleton_coords = list(zip(skeleton_points[0], skeleton_points[1]))
    skeleton_values = adjusted_distances[skeleton]
    
    # 创建结果图像
    custom_skeleton = np.zeros_like(binary_image, dtype=bool)
    
    # 对于每个骨架点，创建其影响区域
    for (y, x), max_dist in zip(skeleton_coords, skeleton_values):
        if max_dist > 0:
            # 创建该点的距离场
            y_indices, x_indices = np.indices(binary_image.shape)
            distances = np.sqrt((y_indices - y)**2 + (x_indices - x)**2)
            
            # 标记在影响范围内的点
            in_range = distances <= max_dist
            custom_skeleton = np.logical_or(custom_skeleton, in_range)
    
    return edt, skeleton, custom_skeleton

# 更高效的版本（使用向量化操作）
def generate_custom_skeleton_efficient(binary_image, a=1):
    """
    高效版本：使用向量化操作生成自定义宽度的血管骨架图
    """
    # 计算距离变换和骨架
    edt = distance_transform_edt(binary_image)
    skeleton = skeletonize(binary_image)
    
    # 获取骨架点处的距离值并应用阈值
    skeleton_distances = np.where(skeleton, edt, 0)
    adjusted_distances = np.minimum(skeleton_distances, a)
    
    # 创建坐标网格
    y_coords, x_coords = np.where(skeleton)
    distances = adjusted_distances[skeleton]
    
    # 为所有骨架点创建影响区域
    custom_skeleton = np.zeros_like(binary_image, dtype=bool)
    
    # 使用广播计算所有点到所有骨架点的距离
    y_grid, x_grid = np.indices(binary_image.shape)
    
    for y, x, max_dist in zip(y_coords, x_coords, distances):
        if max_dist > 0:
            # 计算当前点到所有其他点的距离
            dist_to_point = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
            # 标记在影响范围内的点
            custom_skeleton = np.logical_or(custom_skeleton, dist_to_point <= max_dist)
    
    return edt, skeleton, custom_skeleton

# 使用示例
# image_path = "21.png"
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# if img is None:
#     raise ValueError(f"无法读取图像: {image_path}")

image_path = "21.png"

cap = cv2.VideoCapture(image_path)
ret, img = cap.read()
cap.release()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype(np.uint8)

# 二值化图像
_, binary_img = cv2.threshold(img, 100, 1, cv2.THRESH_BINARY)

# 测试不同a值
a_values = [1, 2, 3, 4]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, a in enumerate(a_values):
    edt, skeleton, custom_skeleton = generate_custom_skeleton_efficient(binary_img, a=a)
    axes[i].imshow(custom_skeleton, cmap='gray')
    axes[i].set_title(f'Custom Skeleton (a={a})')
    axes[i].axis('off')

plt.tight_layout()
plt.show()