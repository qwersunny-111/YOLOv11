import os
import csv

# 定义图片的宽度和高度
img_width = 568
img_height = 528

# YOLO标签文件夹路径
label_folder = '/home/sunleyao/sly/ultralytics/runs/detect/predict2/labels'

# CSV文件路径
output_csv = '/home/sunleyao/sly/ultralytics/output.csv'

# CSV文件的目标行数
target_row_count = 4999

# 初始化CSV文件的表头
header = ['ID', 'image_id', 'class_id', 'x_min', 'y_min', 'width', 'height']

# 初始化ID计数器和行数计数器
id_counter = 1
row_count = 0
data_rows = []

# 读取YOLO标签文件夹中的所有txt文件，获取初始数据
for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        image_id = int(filename.split('.')[0])  # 假设文件名是image_id.txt
        
        # 读取YOLO标签文件
        with open(os.path.join(label_folder, filename), 'r') as file:
            for line in file:
                # 解析YOLO格式
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height 
                
                # 计算宽高的平均值，作为正方形边长
                side_length = int((width + height) / 2)
                x_min = int(x_center - side_length / 2)
                y_min = int(y_center - side_length / 2)
                
                # 将数据添加到行列表
                data_rows.append([id_counter, image_id, class_id, x_min, y_min, side_length, side_length])
                id_counter += 1
                row_count += 1

# 如果读取的数据不足4999行，添加填充行
while row_count < target_row_count:
    # 使用指定的填充格式追加行
    data_rows.append([id_counter, 99999, 9, 0, 0, 0, 0])  # 固定填充值
    id_counter += 1
    row_count += 1

# 写入CSV文件
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)  # 写入表头
    writer.writerows(data_rows[:target_row_count])  # 写入数据行，确保仅写4999行

print(f'转换完成，CSV文件已保存为 {output_csv}，共包含 {target_row_count} 行')
