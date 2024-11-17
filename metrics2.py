import os
from collections import defaultdict
import numpy as np

def parse_yolo_format(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes

def calculate_iou(box1, box2):
    x_left = max(box1[1], box2[1])
    y_top = max(box1[2], box2[2])
    x_right = min(box1[3], box2[3])
    y_bottom = min(box1[4], box2[4])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[3] - box1[1]) * (box1[4] - box1[2])
    box2_area = (box2[3] - box2[1]) * (box2[4] - box2[2])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def compute_precision_recall(true_boxes, pred_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = len(true_boxes)

    detected = set()

    for pred_box in pred_boxes:
        pred_class, *_ = pred_box
        best_iou = 0
        best_j = -1
        for j, true_box in enumerate(true_boxes):
            true_class, *_ = true_box
            if true_class == pred_class and j not in detected:
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

        if best_iou > iou_threshold and best_j != -1:
            true_positives += 1
            false_negatives -= 1
            detected.add(best_j)
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return true_positives, false_positives, false_negatives

def compute_ap(recall, precision):
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def evaluate_detection(ground_truth_folder, predictions_folder, iou_threshold=0.5):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    gt_files = set(os.listdir(ground_truth_folder))
    pred_files = set(os.listdir(predictions_folder))

    # Find missing files
    missing_gt_files = gt_files - pred_files
    missing_pred_files = pred_files - gt_files

    # Process files that are present in both folders
    common_files = gt_files & pred_files

    for gt_file in common_files:
        true_boxes = parse_yolo_format(os.path.join(ground_truth_folder, gt_file))
        pred_boxes = parse_yolo_format(os.path.join(predictions_folder, gt_file))

        if not true_boxes or not pred_boxes:
            continue

        true_positives, false_positives, false_negatives = compute_precision_recall(true_boxes, pred_boxes, iou_threshold)
        
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        
        print(f"File: {gt_file}, True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}")

    # Handle missing prediction files (false negatives)
    for gt_file in missing_gt_files:
        true_boxes = parse_yolo_format(os.path.join(ground_truth_folder, gt_file))
        false_negatives = len(true_boxes)
        
        total_false_negatives += false_negatives
        
        print(f"File: {gt_file}, True Positives: 0, False Positives: 0, False Negatives: {false_negatives}")

    precision = total_true_positives / (total_true_positives + total_false_positives + total_false_negatives) if (total_true_positives + total_false_positives + total_false_negatives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    ap = compute_ap([recall], [precision])

    print(f"\nTotal True Positives: {total_true_positives}")
    print(f"Total False Positives: {total_false_positives}")
    print(f"Total False Negatives: {total_false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@IoU={iou_threshold}: {ap:.4f}")

    return ap

# Example usage:
ground_truth_folder = '/home/B_UserData/sunleyao/WeedDetect/labels4test/labels_txt'
predictions_folder = '/home/B_UserData/sunleyao/WeedDetect/labels'

mean_ap = evaluate_detection(ground_truth_folder, predictions_folder)






