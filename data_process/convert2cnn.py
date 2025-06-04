import os
import cv2
import yaml
import numpy as np
from pathlib import Path

def load_yaml(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"Không tìm thấy file {yaml_path}")
        return None

def get_bboxes_from_label(label_path, class_names):
    bboxes = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            bboxes.append((class_id, x_center, y_center, width, height))
        return bboxes
    except FileNotFoundError:
        print(f"Can not find {label_path}")
        return []

def crop_bbox(image, x_center, y_center, width, height, img_height, img_width):
    # Chuyển đổi tọa độ YOLO (chuẩn hóa) sang pixel
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    cropped_img = image[y_min:y_max, x_min:x_max]

    if cropped_img.size > 0:
        cropped_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
        return cropped_img
    return None

def convert_yolo_to_classification_with_crop(dataset_dir, output_dir, yaml_path):
    data = load_yaml(yaml_path)
    if data is None:
        return
    class_names = data['names']
    splits = ['train']  

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for split in splits:
        img_dir = Path(dataset_dir) / split / 'images'
        label_dir = Path(dataset_dir) / split / 'labels'
        if not img_dir.exists():
            print(f"Unknow folder image {img_dir}, skip...")
            continue
        if not label_dir.exists():
            print(f"Unknow folder label {label_dir}, skip...")
            continue

        split_output_dir = output_dir / split
        split_output_dir.mkdir(exist_ok=True)

        for class_id in range(len(class_names)):
            (split_output_dir / f'class_{class_id}').mkdir(exist_ok=True)

        for img_path in img_dir.glob('*.jpg'): 
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Can not open file {img_path.name}, skip...")
                continue

            img_height, img_width = image.shape[:2]

            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"Can not open {img_path.name}, skip...")
                continue

            # Lấy danh sách bbox
            bboxes = get_bboxes_from_label(label_path, class_names)
            if not bboxes:
                print(f"Label {label_path.name} none, skip...")
                continue

            for idx, (class_id, x_center, y_center, width, height) in enumerate(bboxes):
                cropped_img = crop_bbox(image, x_center, y_center, width, height, img_height, img_width)
                if cropped_img is None:
                    print(f"Bbox {idx} của {img_path.name} không hợp lệ, bỏ qua...")
                    continue

                dest_path = split_output_dir / f'class_{class_id}' / f"{img_path.stem}_bbox{idx}.jpg"
                cv2.imwrite(str(dest_path), cropped_img)
                print(f"Đã lưu ảnh crop {dest_path}")

def main():
    dataset_dir = 'H:/DoAnCV/custom_dataset'  
    output_dir = 'H:/DoAnCV/eff_dataset'  
    yaml_path = 'H:/DoAnCV/custom_dataset/dataset.yml' 

    convert_yolo_to_classification_with_crop(dataset_dir, output_dir, yaml_path)
    print("Done converting YOLO to EfficientNet classification dataset with cropping.")

if __name__ == "__main__":
    main()