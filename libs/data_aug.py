import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import exposure

class RetinaPreprocessor:
    def __init__(self, input_dir, output_base_dir):
        """
        初始化视网膜图像预处理器
        
        参数:
        input_dir: 输入图像目录
        output_base_dir: 输出基础目录
        """
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # 创建输出目录
        self.output_dirs = {
            'original': self.output_base_dir / 'original',
            'grayscale': self.output_base_dir / 'grayscale',
            'normalized': self.output_base_dir / 'normalized',
            'clahe': self.output_base_dir / 'clahe',
            'gamma_corrected': self.output_base_dir / 'gamma_corrected',
            'all_steps': self.output_base_dir / 'all_steps'
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # CLAHE参数
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)
        
        # 伽马校正参数
        self.gamma = 0.5  # gamma < 1 提亮图像，gamma > 1 变暗图像
    
    def convert_to_grayscale(self, image):
        """
        将图像转换为灰度图
        
        参数:
        image: 输入BGR图像
        
        返回:
        灰度图像
        """
        if len(image.shape) == 3:
            # 使用加权平均法进行灰度化（OpenCV默认方法）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray
    
    def normalize_image(self, image):
        """
        图像归一化
        
        参数:
        image: 输入图像
        
        返回:
        归一化后的图像
        """
        # 转换为浮点数以进行归一化
        image_float = image.astype(np.float32)
        
        # 获取最小值和最大值
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        
        # 避免除以零
        if max_val - min_val == 0:
            return image_float.astype(np.uint8)
        
        # 归一化到0-255范围
        # normalized = ((image_float - min_val) / (max_val - min_val)) * 255

        self.mean = 156.2899
        self.std = 26.5457
        img_normalized = (image_float-self.mean)/self.std
        # img_normalized = ((img_normalized - np.min(img_normalized)) 
        #                     / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        # normalized = ((image_float - np.mean(image_float)) / (np.std(image_float) + 1e-8))
        
        return img_normalized.astype(np.uint8)
    
    def apply_clahe(self, image):
        """
        应用对比度受限的自适应直方图均衡化(CLAHE)
        
        参数:
        image: 输入图像
        
        返回:
        CLAHE处理后的图像
        """
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        
        # 应用CLAHE
        clahe_image = clahe.apply(image)
        
        return clahe_image
    
    def apply_gamma_correction(self, image, gamma=None):
        """
        应用伽马校正
        
        参数:
        image: 输入图像
        gamma: 伽马值，如果为None则使用默认值
        
        返回:
        伽马校正后的图像
        """
        if gamma is None:
            gamma = self.gamma
        
        # 归一化到[0, 1]范围
        image_normalized = image.astype(np.float32) / 255.0
        
        # 应用伽马校正
        gamma_corrected = np.power(image_normalized, gamma)
        
        # 缩放回[0, 255]范围并转换为uint8
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        
        return gamma_corrected
    
    def full_pipeline(self, image):
        """
        完整的预处理流程
        
        参数:
        image: 输入图像
        
        返回:
        每一步的处理结果
        """
        results = {}
        
        # 步骤1: 保存原始图像
        results['original'] = image.copy()
        
        # 步骤2: 灰度化
        gray_image = self.convert_to_grayscale(image)
        results['grayscale'] = gray_image
        
        # 步骤3: 归一化
        normalized_image = self.normalize_image(gray_image)
        results['normalized'] = normalized_image
        
        # 步骤4: CLAHE
        clahe_image = self.apply_clahe(normalized_image)
        results['clahe'] = clahe_image
        
        # 步骤5: 伽马校正
        gamma_image = self.apply_gamma_correction(clahe_image)
        results['gamma_corrected'] = gamma_image
        
        return results
    
    def process_single_image(self, image_path):
        """
        处理单张图像并保存所有中间结果
        
        参数:
        image_path: 图像路径
        
        返回:
        处理结果字典
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 获取文件名
        filename = image_path.stem
        
        # 执行完整的预处理流程
        results = self.full_pipeline(image)
        
        # 保存每一步的结果
        for step_name, processed_image in results.items():
            if step_name == 'original' and len(processed_image.shape) == 3:
                # 原始彩色图像需要转换颜色空间
                save_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                save_path = self.output_dirs[step_name] / f"{filename}.png"
                plt.imsave(save_path, save_image)
            else:
                # 灰度图像直接保存
                save_path = self.output_dirs[step_name] / f"{filename}.png"
                cv2.imwrite(str(save_path), processed_image)
        
        # 创建一个包含所有步骤的可视化图像
        self.create_visualization(results, filename)
        
        return results
    
    def create_visualization(self, results, filename):
        """
        创建包含所有预处理步骤的可视化图像
        
        参数:
        results: 处理结果字典
        filename: 文件名
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Retina Image Preprocessing Steps: {filename}', fontsize=16)
        
        steps = [
            ('Original', 'original', False),
            ('Grayscale', 'grayscale', True),
            ('Normalized', 'normalized', True),
            ('CLAHE', 'clahe', True),
            ('Gamma Corrected', 'gamma_corrected', True),
            ('All Steps Combined', 'gamma_corrected', True)  # 最后一步使用伽马校正结果
        ]
        
        for i, (title, key, is_gray) in enumerate(steps):
            ax = axes[i//3, i%3]
            
            if key in results:
                img = results[key]
                if is_gray:
                    ax.imshow(img, cmap='gray')
                else:
                    # 原始图像需要转换颜色空间
                    if len(img.shape) == 3:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img_rgb)
                    else:
                        ax.imshow(img, cmap='gray')
                
                # 添加直方图到每个子图
                if key != 'original' and len(img.shape) == 2:
                    # 在图像下方添加小直方图
                    ax_hist = ax.inset_axes([0.05, -0.25, 0.9, 0.15], transform=ax.transAxes)
                    ax_hist.hist(img.flatten(), bins=50, color='blue', alpha=0.7)
                    ax_hist.set_xlim(0, 255)
                    ax_hist.set_yticks([])
                    ax_hist.set_xticks([0, 128, 255])
            
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dirs['all_steps'] / f"{filename}_visualization.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_directory(self):
        """
        处理输入目录中的所有图像
        """
        # 获取所有支持的图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.ppm']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f'*{ext}'))
            image_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"在 {self.input_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 处理所有图像
        for image_path in tqdm(image_files, desc="处理图像"):
            self.process_single_image(image_path)
        
        print(f"所有处理完成！结果保存在: {self.output_base_dir}")
        self.generate_summary_report()

    def generate_summary_report(self):
        """
        生成处理结果的摘要报告
        """
        report_path = self.output_base_dir / "preprocessing_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("RETINA IMAGE PREPROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Preprocessing Steps Applied:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Grayscale Conversion: Converted color images to grayscale\n")
            f.write("2. Normalization: Scaled pixel values to [0, 255] range\n")
            f.write("3. CLAHE: Contrast Limited Adaptive Histogram Equalization\n")
            f.write(f"   - Clip Limit: {self.clahe_clip_limit}\n")
            f.write(f"   - Grid Size: {self.clahe_grid_size}\n")
            f.write(f"4. Gamma Correction: Gamma value = {self.gamma}\n\n")
            
            f.write("Output Directories:\n")
            f.write("-" * 40 + "\n")
            for step_name, dir_path in self.output_dirs.items():
                num_files = len(list(dir_path.glob("*.png")))
                f.write(f"{step_name.capitalize():15s}: {dir_path} ({num_files} files)\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        print(f"报告已生成: {report_path}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    input_directory = "/home/my/data/CHASE_DB1/train/images"  # 请修改为您的图像目录
    output_directory = "/home/my/data/CHASE_DB1/train/data_aug"  # 请修改为输出目录

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  
    
    # 创建预处理器并运行
    preprocessor = RetinaPreprocessor(input_directory, output_directory)
    preprocessor.process_directory()
    
    # 单张图像处理示例
    # image_path = "path/to/single/image.jpg"
    # results = preprocessor.process_single_image(image_path)