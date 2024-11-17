import os

def convert_to_square_bounding_boxes(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # Calculate the average of width and height
                        avg_size = (width + height) / 2.0
                        
                        # Write the square bounding box to the output file
                        outfile.write(f"{int(class_id)} {x_center} {y_center} {avg_size} {avg_size}\n")

if __name__ == "__main__":
    input_folder = '/home/sunleyao/sly/ultralytics/runs/detect/predict/labels'  # Replace with your input folder path
    output_folder = '/home/sunleyao/sly/ultralytics/runs/detect/predict/square_labels'  # Replace with your desired output folder path
    
    convert_to_square_bounding_boxes(input_folder, output_folder)






