import os
import random
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_yolo_dataset(dataset_path):
    """
    Analyze a YOLO-format dataset with structure: 
    dataset/train/images, dataset/train/labels, dataset/val/images, dataset/val/labels.

    Args:
        dataset_path (str): Path to the YOLO dataset

    Returns:
        dict: Dataset statistics and sample image paths
    """
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
        'sample_images': {}
    }

    # Find YOLO config file (e.g., data.yaml)
    yaml_file = None
    for file in os.listdir(dataset_path):
        if file.endswith('.yaml') or file.endswith('.yml'):
            yaml_file = os.path.join(dataset_path, file)
            break

    if yaml_file:
        import yaml
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            analysis['classes'] = config.get('names', [])

    splits = ['train', 'val']
    for split in splits:
        image_dir = os.path.join(dataset_path, split, 'images')
        label_dir = os.path.join(dataset_path, split, 'labels')

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Warning: Missing {image_dir} or {label_dir}")
            continue

        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
        analysis[split]['total_images'] = len(image_files)

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, c = img.shape
            analysis[split]['image_sizes'].append((w, h))

            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.readlines()

                for label in labels:
                    parts = label.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    analysis[split]['label_distribution'][class_id] += 1
                    analysis[split]['bbox_stats']['total_bboxes'] += 1

                    x_center, y_center, width, height = map(float, parts[1:5])
                    bbox_area = width * height
                    analysis[split]['bbox_stats']['bbox_sizes'].append(bbox_area)

                    if split == 'train' and class_id not in analysis['sample_images']:
                        analysis['sample_images'][class_id] = img_path

        if analysis[split]['total_images'] > 0:
            analysis[split]['bbox_stats']['avg_bboxes_per_image'] = (
                analysis[split]['bbox_stats']['total_bboxes'] / analysis[split]['total_images']
            )

            sizes = np.array(analysis[split]['image_sizes'])
            analysis[split]['image_size_stats'] = {
                'avg_width': np.mean(sizes[:, 0]) if len(sizes) > 0 else 0,
                'avg_height': np.mean(sizes[:, 1]) if len(sizes) > 0 else 0,
                'min_width': np.min(sizes[:, 0]) if len(sizes) > 0 else 0,
                'min_height': np.min(sizes[:, 1]) if len(sizes) > 0 else 0,
                'max_width': np.max(sizes[:, 0]) if len(sizes) > 0 else 0,
                'max_height': np.max(sizes[:, 1]) if len(sizes) > 0 else 0
            }

            if analysis[split]['bbox_stats']['bbox_sizes']:
                analysis[split]['bbox_stats']['avg_bbox_area'] = np.mean(analysis[split]['bbox_stats']['bbox_sizes'])
                analysis[split]['bbox_stats']['min_bbox_area'] = np.min(analysis[split]['bbox_stats']['bbox_sizes'])
                analysis[split]['bbox_stats']['max_bbox_area'] = np.max(analysis[split]['bbox_stats']['bbox_sizes'])

    return analysis

def plot_label_distribution(train_dist, val_dist, classes):
    """Plot class distribution for train and val sets"""
    plt.figure(figsize=(12, 6))
    class_names = [classes[i] if i < len(classes) else f'Class {i}' for i in set(list(train_dist.keys()) + list(val_dist.keys()))]

    train_counts = [train_dist.get(i, 0) for i in range(len(class_names))]
    val_counts = [val_dist.get(i, 0) for i in range(len(class_names))]

    x = np.arange(len(class_names))
    width = 0.35

    plt.bar(x - width/2, train_counts, width, label='Train', color='#1f77b4')
    plt.bar(x + width/2, val_counts, width, label='Val', color='#ff7f0e')

    plt.title('Class Distribution (Train vs Val)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.close()

def main(dataset_path):
    try:
        analysis = analyze_yolo_dataset(dataset_path)

        print("=== YOLO DATASET ANALYSIS ===")
        print("\nTRAIN SET:")
        print(f"Total images: {analysis['train']['total_images']}")
        print("\nImage size stats:")
        print(f"  Avg size: {analysis['train']['image_size_stats']['avg_width']:.2f}x{analysis['train']['image_size_stats']['avg_height']:.2f}")
        print(f"  Min size: {analysis['train']['image_size_stats']['min_width']}x{analysis['train']['image_size_stats']['min_height']}")
        print(f"  Max size: {analysis['train']['image_size_stats']['max_width']}x{analysis['train']['image_size_stats']['max_height']}")
        print("\nBounding box stats:")
        print(f"  Total boxes: {analysis['train']['bbox_stats']['total_bboxes']}")
        print(f"  Avg boxes/image: {analysis['train']['bbox_stats']['avg_bboxes_per_image']:.2f}")
        if analysis['train']['bbox_stats']['bbox_sizes']:
            print(f"  Avg area: {analysis['train']['bbox_stats']['avg_bbox_area']:.4f}")
            print(f"  Min area: {analysis['train']['bbox_stats']['min_bbox_area']:.4f}")
            print(f"  Max area: {analysis['train']['bbox_stats']['max_bbox_area']:.4f}")

        print("\nVAL SET:")
        print(f"Total images: {analysis['val']['total_images']}")
        print("\nImage size stats:")
        print(f"  Avg size: {analysis['val']['image_size_stats']['avg_width']:.2f}x{analysis['val']['image_size_stats']['avg_height']:.2f}")
        print(f"  Min size: {analysis['val']['image_size_stats']['min_width']}x{analysis['val']['image_size_stats']['min_height']}")
        print(f"  Max size: {analysis['val']['image_size_stats']['max_width']}x{analysis['val']['image_size_stats']['max_height']}")
        print("\nBounding box stats:")
        print(f"  Total boxes: {analysis['val']['bbox_stats']['total_bboxes']}")
        print(f"  Avg boxes/image: {analysis['val']['bbox_stats']['avg_bboxes_per_image']:.2f}")
        if analysis['val']['bbox_stats']['bbox_sizes']:
            print(f"  Avg area: {analysis['val']['bbox_stats']['avg_bbox_area']:.4f}")
            print(f"  Min area: {analysis['val']['bbox_stats']['min_bbox_area']:.4f}")
            print(f"  Max area: {analysis['val']['bbox_stats']['max_bbox_area']:.4f}")

        print(f"\nNumber of classes: {len(analysis['classes'])}")
        print("\nClass list:")
        for i, class_name in enumerate(analysis['classes']):
            print(f"  Class {i}: {class_name}")

        print("\nSample images per class (train set):")
        for class_id, img_path in analysis['sample_images'].items():
            class_name = analysis['classes'][class_id] if class_id < len(analysis['classes']) else f'Class {class_id}'
            print(f"  {class_name}: {img_path}")

        plot_label_distribution(analysis['train']['label_distribution'], analysis['val']['label_distribution'], analysis['classes'])
        print("\nLabel distribution plot saved as: label_distribution.png")

    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    dataset_path = "H:/DoAnCV/custom_dataset"
    main(dataset_path)
