import os
import random
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_yolo_dataset(dataset_path):
    """
    Phân tích dataset YOLO với cấu trúc dataset/train/images, dataset/train/labels, dataset/val/images, dataset/val/labels.

    Args:
        dataset_path (str): Đường dẫn tới thư mục dataset YOLO

    Returns:
        dict: Thông tin phân tích dataset và danh sách ảnh mẫu
    """
    # Khởi tạo các biến lưu thông tin
    analysis = {
        'train': {
            'total_images': 0,
            'image_sizes': [],
            'label_distribution': defaultdict(int),
            'bbox_stats': {'total_bboxes': 0, 'avg_bboxes_per_image': 0, 'bbox_sizes': []}
        },
        'val': {
            'total_images': 0,
            'image_sizes': [],
            'label_distribution': defaultdict(int),
            'bbox_stats': {'total_bboxes': 0, 'avg_bboxes_per_image': 0, 'bbox_sizes': []}
        },
        'classes': {},
        'sample_images': {}  # Lưu ảnh mẫu cho mỗi class (từ tập train)
    }

    # Tìm file cấu hình YOLO (thường là data.yaml)
    yaml_file = None
    for file in os.listdir(dataset_path):
        if file.endswith('.yaml') or file.endswith('.yml'):
            yaml_file = os.path.join(dataset_path, file)
            break

    # Đọc file yaml để lấy thông tin class
    if yaml_file:
        import yaml
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            analysis['classes'] = config.get('names', [])

    # Định nghĩa các thư mục
    splits = ['train', 'val']
    for split in splits:
        image_dir = os.path.join(dataset_path, split, 'images')
        label_dir = os.path.join(dataset_path, split, 'labels')

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Cảnh báo: Không tìm thấy thư mục {image_dir} hoặc {label_dir}")
            continue

        # Duyệt qua tất cả các file ảnh
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
        analysis[split]['total_images'] = len(image_files)

        # Phân tích từng ảnh và nhãn
        for img_file in image_files:
            # Đọc ảnh
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Lưu kích thước ảnh
            h, w, c = img.shape
            analysis[split]['image_sizes'].append((w, h))

            # Đọc file nhãn tương ứng
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.readlines()

                # Đếm số lượng bounding box và thống kê class
                for label in labels:
                    parts = label.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    analysis[split]['label_distribution'][class_id] += 1
                    analysis[split]['bbox_stats']['total_bboxes'] += 1

                    # Tính kích thước bounding box (tương đối)
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bbox_area = width * height
                    analysis[split]['bbox_stats']['bbox_sizes'].append(bbox_area)

                    # Lưu ảnh mẫu cho mỗi class (chỉ từ tập train)
                    if split == 'train' and class_id not in analysis['sample_images']:
                        analysis['sample_images'][class_id] = img_path

        # Tính thống kê bổ sung
        if analysis[split]['total_images'] > 0:
            analysis[split]['bbox_stats']['avg_bboxes_per_image'] = (
                analysis[split]['bbox_stats']['total_bboxes'] / analysis[split]['total_images']
            )

            # Tính thống kê kích thước ảnh
            sizes = np.array(analysis[split]['image_sizes'])
            analysis[split]['image_size_stats'] = {
                'avg_width': np.mean(sizes[:, 0]) if len(sizes) > 0 else 0,
                'avg_height': np.mean(sizes[:, 1]) if len(sizes) > 0 else 0,
                'min_width': np.min(sizes[:, 0]) if len(sizes) > 0 else 0,
                'min_height': np.min(sizes[:, 1]) if len(sizes) > 0 else 0,
                'max_width': np.max(sizes[:, 0]) if len(sizes) > 0 else 0,
                'max_height': np.max(sizes[:, 1]) if len(sizes) > 0 else 0
            }

            # Tính thống kê kích thước bounding box
            if analysis[split]['bbox_stats']['bbox_sizes']:
                analysis[split]['bbox_stats']['avg_bbox_area'] = np.mean(analysis[split]['bbox_stats']['bbox_sizes'])
                analysis[split]['bbox_stats']['min_bbox_area'] = np.min(analysis[split]['bbox_stats']['bbox_sizes'])
                analysis[split]['bbox_stats']['max_bbox_area'] = np.max(analysis[split]['bbox_stats']['bbox_sizes'])

    return analysis

def plot_label_distribution(train_dist, val_dist, classes):
    """Vẽ biểu đồ phân bố nhãn cho train và val"""
    plt.figure(figsize=(12, 6))
    class_names = [classes[i] if i < len(classes) else f'Class {i}' for i in set(list(train_dist.keys()) + list(val_dist.keys()))]

    train_counts = [train_dist.get(i, 0) for i in range(len(class_names))]
    val_counts = [val_dist.get(i, 0) for i in range(len(class_names))]

    x = np.arange(len(class_names))
    width = 0.35

    plt.bar(x - width/2, train_counts, width, label='Train', color='#1f77b4')
    plt.bar(x + width/2, val_counts, width, label='Val', color='#ff7f0e')

    plt.title('Phân bố các class trong dataset (Train vs Val)')
    plt.xlabel('Class')
    plt.ylabel('Số lượng')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.close()

def main(dataset_path):
    """Hàm chính để chạy phân tích"""
    try:
        # Phân tích dataset
        analysis = analyze_yolo_dataset(dataset_path)

        # In kết quả phân tích
        print("=== PHÂN TÍCH DATASET YOLO ===")
        print("\nTẬP TRAIN:")
        print(f"Tổng số ảnh: {analysis['train']['total_images']}")
        print("\nThống kê kích thước ảnh:")
        print(f"  Kích thước trung bình: {analysis['train']['image_size_stats']['avg_width']:.2f}x{analysis['train']['image_size_stats']['avg_height']:.2f}")
        print(f"  Kích thước nhỏ nhất: {analysis['train']['image_size_stats']['min_width']}x{analysis['train']['image_size_stats']['min_height']}")
        print(f"  Kích thước lớn nhất: {analysis['train']['image_size_stats']['max_width']}x{analysis['train']['image_size_stats']['max_height']}")
        print("\nThống kê bounding box:")
        print(f"  Tổng số bounding box: {analysis['train']['bbox_stats']['total_bboxes']}")
        print(f"  Số bounding box trung bình mỗi ảnh: {analysis['train']['bbox_stats']['avg_bboxes_per_image']:.2f}")
        if analysis['train']['bbox_stats']['bbox_sizes']:
            print(f"  Diện tích bbox trung bình: {analysis['train']['bbox_stats']['avg_bbox_area']:.4f}")
            print(f"  Diện tích bbox nhỏ nhất: {analysis['train']['bbox_stats']['min_bbox_area']:.4f}")
            print(f"  Diện tích bbox lớn nhất: {analysis['train']['bbox_stats']['max_bbox_area']:.4f}")

        print("\nTẬP VAL:")
        print(f"Tổng số ảnh: {analysis['val']['total_images']}")
        print("\nThống kê kích thước ảnh:")
        print(f"  Kích thước trung bình: {analysis['val']['image_size_stats']['avg_width']:.2f}x{analysis['val']['image_size_stats']['avg_height']:.2f}")
        print(f"  Kích thước nhỏ nhất: {analysis['val']['image_size_stats']['min_width']}x{analysis['val']['image_size_stats']['min_height']}")
        print(f"  Kích thước lớn nhất: {analysis['val']['image_size_stats']['max_width']}x{analysis['val']['image_size_stats']['max_height']}")
        print("\nThống kê bounding box:")
        print(f"  Tổng số bounding box: {analysis['val']['bbox_stats']['total_bboxes']}")
        print(f"  Số bounding box trung bình mỗi ảnh: {analysis['val']['bbox_stats']['avg_bboxes_per_image']:.2f}")
        if analysis['val']['bbox_stats']['bbox_sizes']:
            print(f"  Diện tích bbox trung bình: {analysis['val']['bbox_stats']['avg_bbox_area']:.4f}")
            print(f"  Diện tích bbox nhỏ nhất: {analysis['val']['bbox_stats']['min_bbox_area']:.4f}")
            print(f"  Diện tích bbox lớn nhất: {analysis['val']['bbox_stats']['max_bbox_area']:.4f}")

        print(f"\nSố class: {len(analysis['classes'])}")
        print("\nDanh sách class:")
        for i, class_name in enumerate(analysis['classes']):
            print(f"  Class {i}: {class_name}")

        print("\nẢnh mẫu cho mỗi class (từ tập train):")
        for class_id, img_path in analysis['sample_images'].items():
            class_name = analysis['classes'][class_id] if class_id < len(analysis['classes']) else f'Class {class_id}'
            print(f"  {class_name}: {img_path}")

        # Vẽ biểu đồ phân bố nhãn
        plot_label_distribution(analysis['train']['label_distribution'], analysis['val']['label_distribution'], analysis['classes'])
        print("\nBiểu đồ phân bố nhãn đã được lưu tại: label_distribution.png")

    except Exception as e:
        print(f"Lỗi khi phân tích dataset: {str(e)}")

if __name__ == "__main__":
    # Thay đổi đường dẫn này tới thư mục dataset của bạn
    dataset_path = "H:/DoAnCV/custom_dataset"
    main(dataset_path)