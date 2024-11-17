import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = '/home/B_UserData/sunleyao/WeedDetect/datasets2/train/images'  # 替换为你的输入文件夹路径
output_folder = '/home/B_UserData/sunleyao/WeedDetect/datasets3/train/images'  # 替换为你的输出文件夹路径

# 如果输出文件夹不存在，创建该文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义函数来分离前景和背景
def separate_foreground(image):
    # 将图像从BGR转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 设置HSV颜色阈值来提取绿色（适用于棉七和杂草）
    # 根据具体情况调整这些阈值
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # 创建掩膜：仅保留绿色部分
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 使用掩膜提取前景
    foreground = cv2.bitwise_and(image, image, mask=mask)
    
    # 应用一些形态学操作来去除噪点（可选）
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    
    return foreground

# 遍历输入文件夹中的所有图像
for filename in os.listdir(input_folder):
    # 仅处理图像文件（比如.jpg, .png格式）
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # 分离前景
        foreground = separate_foreground(image)
        
        # 保存分离后的前景图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, foreground)

print("处理完成，所有图像已保存到输出文件夹。")
