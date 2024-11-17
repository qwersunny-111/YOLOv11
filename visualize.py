import os
from PIL import Image, ImageDraw

def parse_yolo_format(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def draw_bounding_boxes(image, boxes, image_size=(568, 528)):
    draw = ImageDraw.Draw(image)
    width, height = image_size
    
    for box in boxes:
        class_id, x_center, y_center, width_box, height_box = box
        
        # Convert normalized coordinates to pixel coordinates
        x_center_pixel = x_center * width
        y_center_pixel = y_center * height
        width_pixel = width_box * width
        height_pixel = height_box * height
        
        # Calculate the corners of the bounding box
        x_min = x_center_pixel - width_pixel / 2
        y_min = y_center_pixel - height_pixel / 2
        x_max = x_center_pixel + width_pixel / 2
        y_max = y_center_pixel + height_pixel / 2
        
        # Ensure the bounding box is within the image boundaries
        x_min = max(0, min(x_min, width))
        y_min = max(0, min(y_min, height))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))
        
        # Determine the color based on class_id
        if class_id == 1:
            outline_color = "red"
        elif class_id == 0:
            outline_color = "blue"
        else:
            outline_color = "green"  # Default color for other class IDs
        
        # Print bounding box coordinates and color for debugging
        print(f"Box: Class ID={class_id}, x_min={x_min:.2f}, y_min={y_min:.2f}, x_max={x_max:.2f}, y_max={y_max:.2f}, Color={outline_color}")
        
        # Draw the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color, width=3)

def visualize_yolo_labels(image_folder, label_folder, output_folder, image_size=(568, 528)):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image and its corresponding label
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_filename = filename
            label_filename = f"{os.path.splitext(filename)[0]}.txt"
            
            image_path = os.path.join(image_folder, image_filename)
            label_path = os.path.join(label_folder, label_filename)
            output_path = os.path.join(output_folder, image_filename)
            
            if os.path.exists(label_path):
                image = Image.open(image_path).resize(image_size)
                boxes = parse_yolo_format(label_path)
                
                draw_bounding_boxes(image, boxes, image_size)
                
                image.save(output_path)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"No corresponding label found for: {image_filename}")

if __name__ == "__main__":
    image_folder = '/home/B_UserData/sunleyao/WeedDetect/weed-detection-new/test/images/'  # Replace with your image folder path
    label_folder = '/home/B_UserData/sunleyao/WeedDetect/labels4test/labels_txt'  # Replace with your label folder path
    output_folder = '/home/B_UserData/sunleyao/WeedDetect/visualize_standard_results'  # Replace with your desired output folder path
    
    visualize_yolo_labels(image_folder, label_folder, output_folder)






