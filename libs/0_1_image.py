from PIL import Image
import numpy as np

def invert_image_pil(image_path, output_path=None):
    """
    使用PIL库反转黑白图像
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
    
    Returns:
        inverted_image: 反转后的图像
    """
    img = Image.open(image_path)
    
    if img.mode != 'L':
        img = img.convert('L')
    
    inverted_img = img.point(lambda x: 255 - x)

    if output_path:
        inverted_img.save(output_path)
        print(f"反转图像已保存到: {output_path}")
    
    return inverted_img

# 使用示例
inverted = invert_image_pil(r'C:\Users\asus\medical-image-segmentation\libs\21.png', 'output_inverted.png')