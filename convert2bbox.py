import json
import cv2
import os

def convert_to_yolo(json_path, img_path, output_txt_path, use_cropped=False):
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return
    try:
        with open(json_path, 'r') as f:
            anno = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return
    img_height, img_width = img.shape[:2]
    
    with open(output_txt_path, 'w') as f:
        for key in anno:
            if key.startswith('item'): 
                try:
                    item = anno[key]
                    category_id = item['category_id'] - 1  # DeepFashion2 bắt đầu từ 1, YOLO từ 0
                    
                    if use_cropped:
                        f.write(f"{category_id} 0.5 0.5 1 1\n")
                    else:
                        bbox = item['bounding_box']  # [x_min, y_min, x_max, y_max]
                        x_min, y_min, x_max, y_max = bbox
                        box_width = x_max - x_min
                        box_height = y_max - y_min
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        
                        # Chuẩn hóa
                        x_center_norm = x_center / img_width
                        y_center_norm = y_center / img_height
                        box_width_norm = box_width / img_width
                        box_height_norm = box_height / img_height
                        
                        if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and
                                0 <= box_width_norm <= 1 and 0 <= box_height_norm <= 1):
                            print(f"Invalid bbox in {json_path}: {bbox}")
                            continue
                        
                        f.write(f"{category_id} {x_center_norm:.6f} {y_center_norm:.6f} {box_width_norm:.6f} {box_height_norm:.6f}\n")
                except Exception as e:
                    print(f"Error processing item {key} in {json_path}: {e}")
                    continue

custom_train_dir = 'custom_dataset/train/images'
custom_val_dir = 'custom_dataset/val/images'
custom_train_anno_dir = 'custom_dataset/train/labels'
custom_val_anno_dir = 'custom_dataset/val/labels'

os.makedirs(custom_train_anno_dir, exist_ok=True)
os.makedirs(custom_val_anno_dir, exist_ok=True)

for img in os.listdir(custom_train_dir):
    if img.endswith('.jpg'):
        json_file = os.path.join(custom_train_anno_dir, img.replace('.jpg', '.json'))
        txt_file = os.path.join(custom_train_anno_dir, img.replace('.jpg', '.txt'))
        img_path = os.path.join(custom_train_dir, img)
        print(f"Processing train image: {img}")
        convert_to_yolo(json_file, img_path, txt_file, use_cropped=False)  # Đặt True nếu dùng ảnh crop

for img in os.listdir(custom_val_dir):
    if img.endswith('.jpg'):
        json_file = os.path.join(custom_val_anno_dir, img.replace('.jpg', '.json'))
        txt_file = os.path.join(custom_val_anno_dir, img.replace('.jpg', '.txt'))
        img_path = os.path.join(custom_val_dir, img)
        print(f"Processing val image: {img}")
        convert_to_yolo(json_file, img_path, txt_file, use_cropped=False)  # Đặt True nếu dùng ảnh crop