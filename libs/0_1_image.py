import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def probability_to_binary(prob_map_path, threshold=0.5, output_path=None, visualize=True):
    """
    将概率图转换为二值图
    
    参数:
    prob_map_path: 概率图文件路径
    threshold: 阈值，默认0.5
    output_path: 输出二值图保存路径，如果为None则不保存
    visualize: 是否显示结果对比图
    
    返回:
    binary_image: 二值图像
    """
    # 1. 读取概率图
    prob_map = cv2.imread(str(prob_map_path), cv2.IMREAD_UNCHANGED)
    
    if prob_map is None:
        raise FileNotFoundError(f"无法读取图像: {prob_map_path}")
    
    print(f"图像形状: {prob_map.shape}")
    print(f"数据类型: {prob_map.dtype}")
    print(f"像素值范围: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # 2. 确保概率图是单通道的
    if len(prob_map.shape) == 3:
        # 如果是三通道，转换为灰度图
        prob_map = cv2.cvtColor(prob_map, cv2.COLOR_BGR2GRAY)
        print("已转换为单通道灰度图")
    
    # 3. 归一化到0-1范围（如果当前范围不是0-1）
    if prob_map.max() > 1.0:
        prob_map_normalized = prob_map.astype(np.float32) / 255.0
        print(f"已将图像归一化到 [0, 1] 范围")
    else:
        prob_map_normalized = prob_map.astype(np.float32)
    
    print(f"归一化后像素值范围: [{prob_map_normalized.min():.3f}, {prob_map_normalized.max():.3f}]")
    
    # 4. 应用阈值转换为二值图
    binary_image = np.where(prob_map_normalized > threshold, 1.0, 0.0).astype(np.float32)
    
    # 5. 转换为8位图像用于保存（0和255）
    binary_image_8bit = (binary_image * 255).astype(np.uint8)
    
    # 6. 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), binary_image_8bit)
        print(f"二值图已保存到: {output_path}")
    
    # 7. 可视化
    if visualize:
        # 创建可视化对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始概率图
        im0 = axes[0].imshow(prob_map_normalized, cmap='gray')
        axes[0].set_title(f'原始概率图\n范围: [{prob_map_normalized.min():.3f}, {prob_map_normalized.max():.3f}]')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 概率图直方图
        axes[1].hist(prob_map_normalized.flatten(), bins=50, color='blue', alpha=0.7)
        axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'阈值={threshold}')
        axes[1].set_title('概率分布直方图')
        axes[1].set_xlabel('概率值')
        axes[1].set_ylabel('像素数量')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 二值结果图
        im2 = axes[2].imshow(binary_image, cmap='gray')
        axes[2].set_title(f'二值分割结果 (阈值={threshold})\n白色像素比例: {binary_image.mean():.2%}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    # 8. 输出统计信息
    print(f"\n统计信息:")
    print(f"阈值: {threshold}")
    print(f"大于阈值的像素比例: {binary_image.mean():.2%}")
    print(f"二值图中白色像素数量: {np.sum(binary_image > 0.5)}")
    print(f"二值图中黑色像素数量: {np.sum(binary_image <= 0.5)}")
    
    return binary_image_8bit


def batch_convert_probability_maps(input_dir, output_dir, threshold=0.5, file_extensions=None):
    """
    批量转换概率图为二值图
    
    参数:
    input_dir: 输入目录（包含概率图）
    output_dir: 输出目录（保存二值图）
    threshold: 阈值
    file_extensions: 文件扩展名列表，默认为常见图像格式
    """
    if file_extensions is None:
        file_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有图像文件
    image_files = []
    for ext in file_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"在 {input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 批量处理
    stats = []
    for i, img_path in enumerate(image_files):
        print(f"\n处理 {i+1}/{len(image_files)}: {img_path.name}")
        
        # 生成输出路径
        output_path = output_dir / f"{img_path.stem}_binary.png"
        
        try:
            # 转换单个图像
            binary_img = probability_to_binary(
                img_path, 
                threshold=threshold, 
                output_path=output_path,
                visualize=False  # 批量处理时不显示可视化
            )
            
            # 收集统计信息
            white_pixel_ratio = np.sum(binary_img > 0) / binary_img.size
            stats.append({
                'filename': img_path.name,
                'white_pixel_ratio': white_pixel_ratio
            })
            
        except Exception as e:
            print(f"处理失败: {img_path.name}, 错误: {e}")
    
    # 输出批量处理统计报告
    if stats:
        print(f"\n{'='*50}")
        print("批量处理统计报告")
        print(f"{'='*50}")
        
        ratios = [s['white_pixel_ratio'] for s in stats]
        print(f"平均白色像素比例: {np.mean(ratios):.2%}")
        print(f"最小白色像素比例: {np.min(ratios):.2%} ({stats[np.argmin(ratios)]['filename']})")
        print(f"最大白色像素比例: {np.max(ratios):.2%} ({stats[np.argmax(ratios)]['filename']})")
        print(f"标准差: {np.std(ratios):.4f}")
        
        # 保存统计报告
        report_path = output_dir / "conversion_report.txt"
        with open(report_path, 'w') as f:
            f.write("概率图转二值图处理报告\n")
            f.write(f"阈值: {threshold}\n")
            f.write(f"处理文件数量: {len(stats)}\n\n")
            
            f.write("各文件统计:\n")
            for stat in stats:
                f.write(f"{stat['filename']}: 白色像素比例={stat['white_pixel_ratio']:.2%}\n")
            
            f.write(f"\n总体统计:\n")
            f.write(f"平均白色像素比例: {np.mean(ratios):.2%}\n")
            f.write(f"最小白色像素比例: {np.min(ratios):.2%}\n")
            f.write(f"最大白色像素比例: {np.max(ratios):.2%}\n")
            f.write(f"标准差: {np.std(ratios):.4f}\n")
        
        print(f"\n统计报告已保存到: {report_path}")


# 使用示例
if __name__ == "__main__":
    # ====================== 示例1: 处理单张概率图 ======================
    print("示例1: 处理单张概率图")
    print("-" * 50)
    
    # 请修改为您的概率图路径
    prob_map_path = r"C:\Users\asus\medical-image-segmentation\libs\a01.png"
    output_path = r"C:\Users\asus\medical-image-segmentation\libs\binary_output.png"
    
    try:
        # 处理单张图像，显示可视化结果
        binary_result = probability_to_binary(
            prob_map_path=prob_map_path,
            threshold=0.5,  # 可以根据需要调整阈值
            output_path=output_path,
            visualize=True
        )
        
        # 可选：尝试不同的阈值
        thresholds_to_try = [0.3, 0.5, 0.7]
        output_dir = Path("C:/Users/asus/medical-image-segmentation/libs")
        for thresh in thresholds_to_try:
            print(f"\n尝试阈值 {thresh}:")
            binary = probability_to_binary(
                prob_map_path=prob_map_path,
                threshold=thresh,
                output_path=output_dir / f"binary_thresh_{thresh}.png",
                visualize=False
            )
            print(f"白色像素比例: {np.sum(binary > 0) / binary.size:.2%}")
            
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    

    

