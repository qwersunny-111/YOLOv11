import os
import cv2
import json
import random
import shutil
import math

def convert_to_yolo_format(json_file, img_width, img_height):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    yolo_annotations = []
    
    for obj in data['shapes']:
        # 将 class_id 为 "weed" 的标签改成 1，其他标签改成 0
        class_id = 1 if obj['label'] == "mq" else 0
        
        # 获取标注的边界框
        points = obj['points']  # 按 [ [x1, y1], [x2, y2] ] 格式提供
        # points = data['shapes'][0]['points']
        # x_center, y_center = points[0]  # 圆心
        # x_radius, y_radius = points[1]   # 圆周上的点

        # # 计算半径
        # radius = abs(y_center - y_radius)
        # diameter = 2 * radius

        # # 转换为 YOLO 格式，归一化
        # x_center /= img_width
        # y_center /= img_height
        # width = diameter / img_width
        # height = diameter / img_height
        
        x_center, y_center = points[0]
        r = math.sqrt((points[0][0]-points[1][0]) * (points[0][0]-points[1][0])+(points[0][1]-points[1][1]) * (points[0][1]-points[1][1]))
        x1, y1 = x_center - r, y_center-r
        x2, y2 = x_center + r, y_center + r
        x_center = x_center / img_width  # 计算中心点x坐标
        y_center = y_center / img_height  # 计算中心点y坐标

        width = (x2 - x1) / img_width  # 计算宽度
        height = (y2 - y1) / img_height  # 计算高度
        
        # 格式化 YOLO 标签
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

def convert_and_save_labels(image_folder, label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, img_file.replace(".png", ".json"))
        output_path = os.path.join(output_folder, img_file.replace(".png", ".txt"))
        
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        if os.path.exists(label_path):
            yolo_annotations = convert_to_yolo_format(label_path, img_width, img_height)
            with open(output_path, 'w') as f:
                f.write("\n".join(yolo_annotations))
            print(f"转换完成: {img_file} -> {output_path}")
        else:
            print(f"标签文件不存在: {label_path}")
    
    return image_files

def split_and_save_dataset(image_files, train_ratio=0.8):
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 创建文件夹结构
    for folder in ["/home/B_UserData/sunleyao/WeedDetect/datasets3/train/images", "/home/B_UserData/sunleyao/WeedDetect/datasets3/train/labels", "/home/B_UserData/sunleyao/WeedDetect/datasets3/val/images", "/home/B_UserData/sunleyao/WeedDetect/datasets3/val/labels"]:
        os.makedirs(folder, exist_ok=True)

    # 将训练集和验证集文件复制到相应文件夹
    for img_file in train_files:
        shutil.copy(os.path.join("/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/train/images", img_file), "/home/B_UserData/sunleyao/WeedDetect/datasets3/train/images")
        shutil.copy(os.path.join("/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/train/correct_labels", img_file.replace(".png", ".txt")), "/home/B_UserData/sunleyao/WeedDetect/datasets3/train/labels")

    for img_file in val_files:
        shutil.copy(os.path.join("/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/train/images", img_file), "/home/B_UserData/sunleyao/WeedDetect/datasets3/val/images")
        shutil.copy(os.path.join("/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/train/correct_labels", img_file.replace(".png", ".txt")), "/home/B_UserData/sunleyao/WeedDetect/datasets3/val/labels")
    
    print("数据集已划分并保存至 'train' 和 'val' 文件夹。")

if __name__ == "__main__":
    image_folder = "/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/test/images"  # 图像文件夹路径
    label_folder = "/home/B_UserData/sunleyao/WeedDetect/labels4test/labels"  # JSON 标签文件夹路径
    output_folder = "/home/B_UserData/sunleyao/WeedDetect/labels4test/labels_txt"  # YOLO 格式标签文件夹路径
    
    # Step 1: 转换标签为 YOLO 格式
    image_files = convert_and_save_labels(image_folder, label_folder, output_folder)
    
    # # Step 2: 划分数据集并保存到 'train' 和 'val' 文件夹
    # split_and_save_dataset(image_files, train_ratio=0.8)
