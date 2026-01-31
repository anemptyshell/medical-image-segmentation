import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple
import os

class U_Net_Attention_Visualizer:
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        初始化U-Net注意力可视化器
        
        Args:
            model: 训练好的U-Net模型
            device: 设备 (cpu或cuda)
        """
        self.model = model
        self.model.eval()  # 设置为评估模式
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 注册钩子来捕获d2层的特征图
        self.d2_features = None
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子来捕获最后一个卷积层的特征图"""
        def get_features_hook(module, input, output):
            self.d2_features = output.detach().cpu()
        
        # 找到最后一个卷积层（Up_conv2）
        for name, module in self.model.named_modules():
            if name == 'Up_conv2':
                module.register_forward_hook(get_features_hook)
                break
    
    def compute_channel_attention(self, feature_map: torch.Tensor, method: str = 'mean') -> np.ndarray:
        """
        计算通道注意力热图
        
        Args:
            feature_map: 特征图 [batch, channels, height, width]
            method: 计算方法 - 'mean', 'max', 'std', 'l2_norm'
        
        Returns:
            注意力热图 [height, width]
        """
        # 确保是单个样本
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]  # 取第一个样本
        
        feature_map = feature_map.numpy()
        
        if method == 'mean':
            # 方法1：通道平均
            attention_map = np.mean(feature_map, axis=0)
        elif method == 'max':
            # 方法2：通道最大值
            attention_map = np.max(feature_map, axis=0)
        elif method == 'std':
            # 方法3：通道标准差
            attention_map = np.std(feature_map, axis=0)
        elif method == 'l2_norm':
            # 方法4：通道L2范数
            attention_map = np.linalg.norm(feature_map, ord=2, axis=0)
        elif method == 'grad_cam':
            # 方法5：类似Grad-CAM的方法
            weights = np.mean(feature_map, axis=(1, 2))  # 全局平均池化作为权重
            attention_map = np.zeros_like(feature_map[0])
            for i in range(feature_map.shape[0]):
                attention_map += weights[i] * feature_map[i]
        else:
            raise ValueError(f"未知的注意力计算方法: {method}")
        
        return attention_map
    
    def compute_spatial_attention(self, feature_map: torch.Tensor) -> np.ndarray:
        """
        计算空间注意力热图
        
        Args:
            feature_map: 特征图 [batch, channels, height, width]
        
        Returns:
            空间注意力热图 [height, width]
        """
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]
        
        feature_map = feature_map.numpy()
        
        # 计算每个位置所有通道的激活强度
        spatial_attention = np.linalg.norm(feature_map, ord=2, axis=0)
        
        return spatial_attention
    
    def normalize_attention_map(self, attention_map: np.ndarray) -> np.ndarray:
        """
        标准化注意力热图
        
        Args:
            attention_map: 原始注意力热图
        
        Returns:
            标准化后的热图 [0, 1]
        """
        # 归一化到0-1
        if attention_map.max() > attention_map.min():
            normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        else:
            normalized = np.zeros_like(attention_map)
        
        return normalized
    
    def overlay_heatmap(self, original_image: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        将热图叠加到原始图像上

        Args:
            original_image: 原始图像 [H, W, C] or [H, W]
            heatmap: 热图 [H, W]
            alpha: 叠加透明度
            colormap: OpenCV颜色映射

        Returns:
            叠加后的图像
        """
        # 确保热图是float32类型并归一化到0-1
        heatmap = heatmap.astype(np.float32)

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        # 将热图转换为uint8 (0-255)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        # 将热图转换为彩色
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # 确保原始图像是彩色且数据类型为uint8
        if len(original_image.shape) == 2:
            # 灰度图转彩色
            if original_image.dtype != np.uint8:
                if original_image.max() <= 1.0:
                    original_image = (original_image * 255).astype(np.uint8)
                else:
                    original_image = original_image.astype(np.uint8)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        elif original_image.shape[2] == 1:
            # 单通道转彩色
            if original_image.dtype != np.uint8:
                if original_image.max() <= 1.0:
                    original_image = (original_image * 255).astype(np.uint8)
                else:
                    original_image = original_image.astype(np.uint8)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            # 已经是彩色图，确保数据类型
            if original_image.dtype != np.uint8:
                if original_image.max() <= 1.0:
                    original_image = (original_image * 255).astype(np.uint8)
                else:
                    original_image = original_image.astype(np.uint8)

        # 确保热图大小匹配
        if heatmap_colored.shape[:2] != original_image.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, 
                                        (original_image.shape[1], original_image.shape[0]))

        # 确保两个图像都是相同的uint8类型
        heatmap_colored = heatmap_colored.astype(np.uint8)
        original_image = original_image.astype(np.uint8)

        # 叠加
        overlay = cv2.addWeighted(original_image, 1 - alpha, 
                                 heatmap_colored, alpha, 0)

        return overlay
    
    def visualize(self, image: torch.Tensor, 
              method: str = 'mean',
              save_path: Optional[str] = None,
              figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        完整的可视化流程

        Args:
            image: 输入图像 [batch, channels, height, width]
            method: 注意力计算方法
            save_path: 保存路径（如果不保存则为None）
            figsize: 图像大小
        """
        # 清空前一个特征图
        self.d2_features = None

        # 前向传播
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)

        # 获取特征图
        if self.d2_features is None:
            raise RuntimeError("未能捕获d2层特征图，请检查钩子注册")

        # 计算注意力热图
        attention_map = self.compute_channel_attention(self.d2_features, method)

        # 标准化
        attention_map_normalized = self.normalize_attention_map(attention_map)

        # 准备原始图像
        if len(image.shape) == 4:
            img_np = image[0].cpu().numpy()  # 取第一个样本
        else:
            img_np = image.cpu().numpy()

        # 处理图像用于显示
        if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3]:
            # 从CHW转HWC
            img_np = np.transpose(img_np, (1, 2, 0))

            # 如果是3通道，转为灰度图用于热图叠加
            if img_np.shape[2] == 3:
                # 保存彩色版本用于显示
                img_color = img_np.copy()

                # 转为灰度用于叠加
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:
                        img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    else:
                        img_gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                # 单通道
                img_gray = img_np.squeeze()
                if len(img_gray.shape) == 2:
                    # 如果是单通道但形状是(H,W,1)，转换为(H,W)
                    if img_gray.shape[2] == 1:
                        img_gray = img_gray[:, :, 0]
                # 创建彩色版本用于显示
                img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        else:
            # 其他情况
            img_gray = img_np
            if len(img_gray.shape) == 2:
                img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            else:
                img_color = img_gray

        # 将灰度图归一化到0-255的uint8
        if img_gray.dtype != np.uint8:
            if img_gray.max() <= 1.0:
                img_gray = (img_gray * 255).astype(np.uint8)
            else:
                img_gray = img_gray.astype(np.uint8)

        # 调整热图大小以匹配原始图像
        if attention_map_normalized.shape != img_gray.shape:
            attention_map_resized = cv2.resize(attention_map_normalized, 
                                              (img_gray.shape[1], img_gray.shape[0]))
        else:
            attention_map_resized = attention_map_normalized

        # 创建叠加图像
        overlay = self.overlay_heatmap(img_gray, attention_map_resized, alpha=0.6)

        # 可视化
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 原始彩色图像
        if img_color.dtype != np.uint8:
            if img_color.max() <= 1.0:
                img_color_display = (img_color * 255).astype(np.uint8)
            else:
                img_color_display = img_color.astype(np.uint8)
        else:
            img_color_display = img_color

        axes[0, 0].imshow(cv2.cvtColor(img_color_display, cv2.COLOR_BGR2RGB) 
                          if img_color_display.shape[2] == 3 else img_color_display, 
                          cmap='gray' if img_color_display.shape[2] == 1 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 原始注意力热图
        im1 = axes[0, 1].imshow(attention_map_normalized, cmap='hot')
        axes[0, 1].set_title(f'Attention Heatmap ({method})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        # 调整大小的热图
        im2 = axes[0, 2].imshow(attention_map_resized, cmap='hot')
        axes[0, 2].set_title(f'Resized Heatmap')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])

        # 叠加图像
        axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Overlay (JET colormap)')
        axes[1, 0].axis('off')

        # 模型输出
        if len(output.shape) == 4:
            output_np = output[0, 0].cpu().numpy()  # 取第一个样本，第一个通道
        else:
            output_np = output.cpu().numpy()

        axes[1, 1].imshow(output_np, cmap='gray')
        axes[1, 1].set_title('Model Output')
        axes[1, 1].axis('off')

        # 热图的3D可视化（可选）
        from mpl_toolkits.mplot3d import Axes3D
        ax3d = fig.add_subplot(2, 3, 6, projection='3d')

        # 降采样以加速3D绘制
        step = max(1, attention_map_resized.shape[0] // 50)
        y, x = np.mgrid[0:attention_map_resized.shape[0]:step, 
                       0:attention_map_resized.shape[1]:step]
        z = attention_map_resized[::step, ::step]

        ax3d.plot_surface(x, y, z, cmap='hot', alpha=0.8)
        ax3d.set_title('3D Attention Surface')
        ax3d.set_xlabel('Width')
        ax3d.set_ylabel('Height')
        ax3d.set_zlabel('Attention')

        plt.tight_layout()

        if save_path:
            # 确保保存路径的目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")

        plt.show()

        return attention_map_normalized
    
    def visualize_multiple_methods(self, image: torch.Tensor, 
                                  methods: list = ['mean', 'max', 'std', 'l2_norm', 'grad_cam'],
                                  save_path: Optional[str] = None) -> None:
        """
        使用多种方法可视化注意力热图
        
        Args:
            image: 输入图像
            methods: 方法列表
            save_path: 保存路径
        """
        # 清空前一个特征图
        self.d2_features = None
        
        # 前向传播
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
        
        # 获取特征图
        if self.d2_features is None:
            raise RuntimeError("未能捕获d2层特征图，请检查钩子注册")
        
        # 准备原始图像
        if len(image.shape) == 4:
            img_np = image[0].cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        if len(img_np.shape) == 3 and img_np.shape[0] == 3:
            img_display = np.transpose(img_np, (1, 2, 0))
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2GRAY)
        elif len(img_np.shape) == 3 and img_np.shape[0] == 1:
            img_display = img_np[0]
        else:
            img_display = img_np
        
        n_methods = len(methods)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(5 * (n_methods + 1), 10))
        
        # 显示原始图像
        axes[0, 0].imshow(img_display, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 显示模型输出
        output_np = output[0, 0].cpu().numpy()
        axes[1, 0].imshow(output_np, cmap='gray')
        axes[1, 0].set_title('Model Output')
        axes[1, 0].axis('off')
        
        # 对每种方法计算和显示热图
        for idx, method in enumerate(methods):
            # 计算热图
            attention_map = self.compute_channel_attention(self.d2_features, method)
            attention_map_normalized = self.normalize_attention_map(attention_map)
            
            # 调整大小
            if attention_map_normalized.shape != img_display.shape:
                attention_map_resized = cv2.resize(attention_map_normalized, 
                                                  (img_display.shape[1], img_display.shape[0]))
            else:
                attention_map_resized = attention_map_normalized
            
            # 叠加
            overlay = self.overlay_heatmap(img_display, attention_map_resized, alpha=0.6)
            
            # 显示热图
            im = axes[0, idx + 1].imshow(attention_map_resized, cmap='hot')
            axes[0, idx + 1].set_title(f'{method}')
            axes[0, idx + 1].axis('off')
            plt.colorbar(im, ax=axes[0, idx + 1])
            
            # 显示叠加图
            axes[1, idx + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1, idx + 1].set_title(f'{method} Overlay')
            axes[1, idx + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()


# 使用示例
# if __name__ == "__main__":
#     # 假设你已经有了训练好的U-Net模型
#     model = U_Net(input_ch=3, output_ch=1)
    
#     # 加载模型权重（如果有）
#     # model.load_state_dict(torch.load('path_to_your_model.pth'))
    
#     # 创建可视化器
#     visualizer = U_Net_Attention_Visualizer(model)
    
#     # 准备一个测试图像
#     batch_size = 1
#     test_image = torch.randn(batch_size, 3, 512, 512)  # 随机测试图像
    
#     # 方法1：使用单一方法可视化
#     print("使用均值方法可视化注意力热图...")
#     attention_map = visualizer.visualize(test_image, method='mean', 
#                                          save_path='attention_mean.png')
    
#     # 方法2：使用多种方法比较
#     print("\n使用多种方法比较注意力热图...")
#     visualizer.visualize_multiple_methods(test_image, 
#                                          methods=['mean', 'max', 'std', 'l2_norm'],
#                                          save_path='attention_comparison.png')
    
    # # 方法3：也可以单独获取热图进行进一步处理
    # with torch.no_grad():
    #     test_image = test_image.to(visualizer.device)
    #     _ = visualizer.model(test_image)
    
    # if visualizer.d2_features is not None:
    #     # 计算不同方法的注意力图
    #     attention_mean = visualizer.compute_channel_attention(visualizer.d2_features, 'mean')
    #     attention_max = visualizer.compute_channel_attention(visualizer.d2_features, 'max')
        
    #     # 标准化
    #     attention_mean_norm = visualizer.normalize_attention_map(attention_mean)
    #     attention_max_norm = visualizer.normalize_attention_map(attention_max)
        
    #     print(f"均值注意力图形状: {attention_mean_norm.shape}")
    #     print(f"最大值注意力图形状: {attention_max_norm.shape}")
        
    #     # 可以保存热图为numpy文件
    #     np.save('attention_mean.npy', attention_mean_norm)
    #     np.save('attention_max.npy', attention_max_norm)