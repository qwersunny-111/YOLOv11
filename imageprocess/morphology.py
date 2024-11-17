import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = '/home/B_UserData/sunleyao/WeedDetect/datasets3/train/images'  # 替换为你的输入文件夹路径
output_folder = '/home/B_UserData/sunleyao/WeedDetect/datasets4/train/images'  # 替换为你的输出文件夹路径

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置内核大小，进行形态学处理
kernel = np.ones((5, 5), np.uint8)

# 遍历输入文件夹中的每个图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # 只处理图片文件
        # 构建完整的输入和输出文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取彩色图像
        image = cv2.imread(input_path)

        # 使用闭运算填充小的黑色空洞
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # 进行膨胀操作，以增强圆形叶片和尖形叶片的轮廓
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # 接着使用腐蚀操作，去除多余的膨胀并保留叶片形状
        processed_image = cv2.erode(dilated, kernel, iterations=1)

        # 保存处理后的图像到输出文件夹
        cv2.imwrite(output_path, processed_image)

        print(f"处理完成: {filename}")

print("所有图像处理完成。")
