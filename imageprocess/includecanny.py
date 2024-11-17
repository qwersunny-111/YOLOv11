import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = '/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/test/images'  # 替换为你的输入文件夹路径
output_folder = '/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/test/images_process'  # 替换为你的输出文件夹路径

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置HSV颜色阈值
lower_green = np.array([35, 40, 40])   # 根据具体需要调整
upper_green = np.array([85, 255, 255])

# 设置形态学内核
kernel = np.ones((5, 5), np.uint8)

# 小块噪声的面积阈值
min_area = 100  # 根据图像调整此值，较小的区域将被去掉

# 遍历输入文件夹中的每个图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # 只处理图片文件
        # 构建完整的输入和输出文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取图像并转换到 HSV 色彩空间
        image = cv2.imread(input_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 阈值分割获取绿色区域
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 使用闭运算填充小的黑色空洞
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 进行开运算去除小块噪声
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # 膨胀操作，以增强圆形叶片和尖形叶片的轮廓
        dilated = cv2.dilate(opened, kernel, iterations=1)
        
        # 边缘检测
        edges = cv2.Canny(dilated, threshold1=60, threshold2=150)

        # 将边缘检测结果与膨胀后的图像结合
        combined = cv2.bitwise_or(dilated, edges)

        # 去除小面积噪声块
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(combined)

        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            if area >= min_area:
                # 保留面积大于阈值的轮廓
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # 创建彩色结果，叠加边缘信息
        result = cv2.bitwise_and(image, image, mask=filtered_mask)

        # 保存处理后的图像到输出文件夹
        cv2.imwrite(output_path, result)

        print(f"处理完成: {filename}")

print("所有图像处理完成。")
