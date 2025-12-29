import os
import shutil

def copy_images_sequentially(source_dir, copies=10):
    """
    将目录中的图片复制多份，按数字顺序命名
    
    Args:
        source_dir: 图片目录路径
        copies: 要复制的份数
    """
    # 检查目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误：目录不存在 - {source_dir}")
        return
    
    # 获取所有jpg文件并按数字排序
    image_files = []
    for f in os.listdir(source_dir):
        if f.lower().endswith('.png'):
            try:
                # 提取文件名中的数字部分
                num = int(''.join(filter(str.isdigit, f)))
                image_files.append((num, f))
            except ValueError:
                # 如果文件名中没有数字，跳过
                continue
    
    if not image_files:
        print("目录中没有找到格式为'数字.jpg'的图片")
        return
    
    # 按数字排序
    image_files.sort(key=lambda x: x[0])
    
    # 获取原图片数量和最大编号
    original_count = len(image_files)
    max_num = image_files[-1][0]  # 最大编号
    
    print(f"找到 {original_count} 张图片，最大编号: {max_num}")
    print(f"原图片: {[f[1] for f in image_files]}")
    
    # 复制文件
    total_copied = 0
    
    for copy_num in range(copies):
        # 计算当前批次的起始编号
        start_num = max_num + 1 + (copy_num * original_count)
        
        print(f"\n正在复制第 {copy_num + 1} 份 (编号 {start_num}~{start_num + original_count - 1}):")
        
        for idx, (orig_num, filename) in enumerate(image_files):
            # 原文件路径
            src_path = os.path.join(source_dir, filename)
            
            # 新文件名
            new_num = start_num + idx
            new_filename = f"{new_num}.png"
            dst_path = os.path.join(source_dir, new_filename)
            
            # 检查目标文件是否已存在
            if os.path.exists(dst_path):
                print(f"  跳过: {new_filename} 已存在")
                continue
            
            # 复制文件
            try:
                shutil.copy2(src_path, dst_path)  # copy2保留元数据
                print(f"  复制: {filename} -> {new_filename}")
                total_copied += 1
            except Exception as e:
                print(f"  错误: 复制 {filename} 失败 - {e}")
    
    # 统计结果
    print(f"\n{'='*50}")
    print("复制完成!")
    print(f"原图片数量: {original_count}")
    print(f"复制份数: {copies}")
    print(f"理论新增文件数: {original_count * copies}")
    print(f"实际复制文件数: {total_copied}")
    
    # 显示新的编号范围
    final_max_num = max_num + (copies * original_count)
    print(f"新的文件编号范围: 1.png ~ {final_max_num}.png")
    
    # 显示各批次编号
    print("\n各批次编号范围:")
    for i in range(copies):
        start = max_num + 1 + (i * original_count)
        end = start + original_count - 1
        print(f"  第{i+1}份: {start}.png ~ {end}.png")

# 使用示例
if __name__ == "__main__":
    # 设置目录路径
    source_directory = r"G:\CHASE_DB1\strong_2"
    
    # 复制10份
    copy_images_sequentially(source_directory, copies=10)