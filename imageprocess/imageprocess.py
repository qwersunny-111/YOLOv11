import cv2
import numpy as np
import os

# 配置路径
input_folder = '/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/test/images'  # 输入图像文件夹
output_folder = '/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/test/images_process'  # 输出图像文件夹

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def threshold_segmentation(image):
    """
    基于HSV颜色空间进行简单的阈值分割，提取棉七和杂草区域并返回掩码
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 棉七颜色阈值（绿色区域），根据实际需要调整
    lower_green = np.array([35, 50, 50])  # 棉七绿色的下限
    upper_green = np.array([85, 255, 255])  # 棉七绿色的上限
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    return mask

def apply_mask(image, mask):
    """
    应用掩码到图像，将背景区域清除
    """
    # 将掩码应用到原始图像，将背景变为黑色
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def process_image(image_path):
    """
    加载和处理图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    # 颜色分割，得到掩码
    mask = threshold_segmentation(image)
    
    # 使用掩码将植物区域保留，背景变为黑色
    processed_image = apply_mask(image, mask)
    
    return processed_image

def save_image(image, output_path):
    """
    保存处理后的图像
    """
    cv2.imwrite(output_path, image)

# 处理数据集中的所有图像
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, image_name)
        
        # 处理图像
        processed_image = process_image(input_path)
        
        # 保存处理结果（只保存去除背景后的图像）
        save_image(processed_image, os.path.join(output_folder, f"{image_name}"))
        
        print(f"{image_name}")

print("预处理完成！")
