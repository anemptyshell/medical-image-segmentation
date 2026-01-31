import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class FeatureMapVisualizer:
    """特征图可视化类"""
    def __init__(self, save_dir="./feature_maps"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def pool_feature_map(self, feature_map, method='max'):
        """
        对特征图进行通道维度池化
        
        Args:
            feature_map: [C, H, W] 或 [1, C, H, W]
            method: 'max' 或 'avg'
            
        Returns:
            池化后的特征图 [H, W]
        """
        # 确保特征图是3D [C, H, W]
        if feature_map.dim() == 4:
            feature_map = feature_map.squeeze(0)  # [C, H, W]
        
        if method == 'max':
            # 通道维度最大池化
            pooled = torch.max(feature_map, dim=0)[0]  # [H, W]
        elif method == 'avg':
            # 通道维度平均池化
            pooled = torch.mean(feature_map, dim=0)  # [H, W]
        else:
            raise ValueError(f"不支持的池化方法: {method}")
        
        return pooled
    
    def normalize_feature_map(self, feature_map):
        """归一化特征图到[0,1]范围"""
        min_val = feature_map.min()
        max_val = feature_map.max()
        if max_val > min_val:
            normalized = (feature_map - min_val) / (max_val - min_val)
        else:
            normalized = feature_map
        return normalized
    
    def visualize_s_features(self, s1, s2, s3, s4, 
                           image_idx=0, 
                           method='max',
                           save=True,
                           show=True):
        """
        可视化s1-s4特征图
        
        Args:
            s1, s2, s3, s4: 特征图张量
            image_idx: 图像索引（用于命名）
            method: 池化方法 'max' 或 'avg'
            save: 是否保存图像
            show: 是否显示图像
        """
        # 池化特征图
        s1_pooled = self.pool_feature_map(s1, method)
        s2_pooled = self.pool_feature_map(s2, method)
        s3_pooled = self.pool_feature_map(s3, method)
        s4_pooled = self.pool_feature_map(s4, method)
        
        # 归一化
        # s1_norm = self.normalize_feature_map(s1_pooled)
        # s2_norm = self.normalize_feature_map(s2_pooled)
        # s3_norm = self.normalize_feature_map(s3_pooled)
        # s4_norm = self.normalize_feature_map(s4_pooled)
        
        # 转换为numpy
        s1_np = s1_pooled.detach().cpu().numpy()
        s2_np = s2_pooled.detach().cpu().numpy()
        s3_np = s3_pooled.detach().cpu().numpy()
        s4_np = s4_pooled.detach().cpu().numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Feature Maps Visualization (Image {image_idx}, Method: {method})', fontsize=14)
        
        # s1 特征图
        im1 = axes[0, 0].imshow(s1_np, cmap='viridis')
        axes[0, 0].set_title(f's1: {s1.shape[1:]} -> {s1_np.shape}')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # s2 特征图
        im2 = axes[0, 1].imshow(s2_np, cmap='viridis')
        axes[0, 1].set_title(f's2: {s2.shape[1:]} -> {s2_np.shape}')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # s3 特征图
        im3 = axes[1, 0].imshow(s3_np, cmap='viridis')
        axes[1, 0].set_title(f's3: {s3.shape[1:]} -> {s3_np.shape}')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # s4 特征图
        im4 = axes[1, 1].imshow(s4_np, cmap='viridis')
        axes[1, 1].set_title(f's4: {s4.shape[1:]} -> {s4_np.shape}')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f"features_image_{image_idx:04d}_{method}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特征图已保存至: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, (s1_np, s2_np, s3_np, s4_np)
    
    def visualize_features_comparison(self, s1, s2, s3, s4, 
                                    image_idx=0,
                                    save=True):
        """
        同时可视化最大池化和平均池化的特征图
        
        Args:
            s1, s2, s3, s4: 特征图张量
            image_idx: 图像索引
            save: 是否保存图像
        """
        fig, axes = plt.subplots(4, 2, figsize=(10, 16))
        fig.suptitle(f'Feature Maps Comparison (Image {image_idx})', fontsize=14)
        
        # 定义池化方法
        methods = ['max', 'avg']
        cmap = 'viridis'
        
        for row_idx, (s_feat, s_name) in enumerate(zip([s1, s2, s3, s4], ['s1', 's2', 's3', 's4'])):
            for col_idx, method in enumerate(methods):
                # 池化特征图
                s_pooled = self.pool_feature_map(s_feat, method)
                s_norm = self.normalize_feature_map(s_pooled)
                s_np = s_norm.detach().cpu().numpy()
                
                # 绘制
                im = axes[row_idx, col_idx].imshow(s_np, cmap=cmap)
                axes[row_idx, col_idx].set_title(f'{s_name} ({method})')
                axes[row_idx, col_idx].axis('off')
                plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f"features_comparison_image_{image_idx:04d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特征图对比已保存至: {save_path}")
        
        plt.show()
    
    def visualize_feature_statistics(self, s1, s2, s3, s4, 
                                   image_idx=0,
                                   save=True):
        """
        可视化特征图的统计信息
        
        Args:
            s1, s2, s3, s4: 特征图张量
            image_idx: 图像索引
            save: 是否保存图像
        """
        # 计算统计信息
        stats = []
        names = ['s1', 's2', 's3', 's4']
        
        for name, feat in zip(names, [s1, s2, s3, s4]):
            if feat.dim() == 4:
                feat = feat.squeeze(0)  # [C, H, W]
            
            # 统计信息
            mean_val = feat.mean().item()
            std_val = feat.std().item()
            max_val = feat.max().item()
            min_val = feat.min().item()
            median_val = feat.median().item()
            
            stats.append({
                'name': name,
                'shape': feat.shape,
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'min': min_val,
                'median': median_val
            })
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Feature Maps Statistics (Image {image_idx})', fontsize=14)
        
        # 1. 均值比较
        names = [s['name'] for s in stats]
        means = [s['mean'] for s in stats]
        axes[0, 0].bar(names, means, color=['blue', 'green', 'red', 'purple'])
        axes[0, 0].set_title('Mean Value per Feature Map')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 标准差比较
        stds = [s['std'] for s in stats]
        axes[0, 1].bar(names, stds, color=['blue', 'green', 'red', 'purple'])
        axes[0, 1].set_title('Standard Deviation per Feature Map')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 极值比较
        max_vals = [s['max'] for s in stats]
        min_vals = [s['min'] for s in stats]
        x = range(len(names))
        axes[1, 0].bar(x, max_vals, width=0.4, label='Max', color='blue', align='center')
        axes[1, 0].bar(x, min_vals, width=0.4, label='Min', color='red', align='edge')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(names)
        axes[1, 0].set_title('Min/Max Values per Feature Map')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 中位数比较
        medians = [s['median'] for s in stats]
        axes[1, 1].bar(names, medians, color=['blue', 'green', 'red', 'purple'])
        axes[1, 1].set_title('Median Value per Feature Map')
        axes[1, 1].set_ylabel('Median')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f"features_stats_image_{image_idx:04d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特征统计图已保存至: {save_path}")
        
        plt.show()
        
        # 打印统计信息
        print("\n=== 特征图统计信息 ===")
        for stat in stats:
            print(f"\n{stat['name']} {stat['shape']}:")
            print(f"  均值: {stat['mean']:.4f}")
            print(f"  标准差: {stat['std']:.4f}")
            print(f"  最大值: {stat['max']:.4f}")
            print(f"  最小值: {stat['min']:.4f}")
            print(f"  中位数: {stat['median']:.4f}")
        
        return stats


# 修改后的test_epoch函数
# def test_epoch(self, test_loader, visualize_features=True, vis_num=5, vis_method='max'):
#     """
#     测试epoch
    
#     Args:
#         test_loader: 测试数据加载器
#         visualize_features: 是否可视化特征图
#         vis_num: 可视化前N张图像的特征图
#         vis_method: 可视化方法 'max' 或 'avg'
#     """
#     self.network.eval()
    
#     # 初始化特征可视化器（如果启用可视化）
#     if visualize_features:
#         visualizer = FeatureMapVisualizer(save_dir="./test_feature_maps")
    
#     with torch.no_grad():
#         for iter, data in enumerate(tqdm(test_loader)):
#             images, targets = data
#             images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

#             preds, edge_pred, loss_mi1, loss_mi2, loss_mi3, loss_mi4, s1,s2,s3,s4,u1,u2,u3,u4 = self.network(images)
            
#             # 可视化特征图（仅对前vis_num个样本）
#             if visualize_features and iter < vis_num:
#                 print(f"\n可视化第 {iter+1}/{vis_num} 张图像的特征图...")
                
#                 # 方法1：基础可视化
#                 visualizer.visualize_s_features(
#                     s1, s2, s3, s4,
#                     image_idx=iter,
#                     method=vis_method,
#                     save=True,
#                     show=True  # 设置为False可加速批量可视化
#                 )
                
#                 # 方法2：对比可视化
#                 visualizer.visualize_features_comparison(
#                     s1, s2, s3, s4,
#                     image_idx=iter,
#                     save=True
#                 )
                
#                 # 方法3：统计信息可视化
#                 visualizer.visualize_feature_statistics(
#                     s1, s2, s3, s4,
#                     image_idx=iter,
#                     save=True
#                 )
                
#                 # 额外的：保存原始特征图数据（用于进一步分析）
#                 feature_data = {
#                     's1': s1.detach().cpu(),
#                     's2': s2.detach().cpu(),
#                     's3': s3.detach().cpu(),
#                     's4': s4.detach().cpu(),
#                     'image_idx': iter
#                 }
                
#                 save_path = os.path.join(visualizer.save_dir, f"feature_data_image_{iter:04d}.pt")
#                 torch.save(feature_data, save_path)
#                 print(f"特征数据已保存至: {save_path}")
    
#     print(f"\n测试完成！特征图已保存至: {visualizer.save_dir if visualize_features else 'N/A'}")



# 使用示例
# if __name__ == "__main__":
#     # 假设您有一个测试类的实例
#     tester = YourTestClass(network, ...)
    
#     # 测试并可视化前3张图像的特征图（使用最大池化）
#     tester.test_epoch(
#         test_loader, 
#         visualize_features=True,
#         vis_num=3,
#         vis_method='max'
#     )
    
    # 或者只测试不可视化
    # tester.test_epoch(test_loader, visualize_features=False)