import os
import random
import shutil

train_image_dir = 'DeepFashion2/deepfashion2_original_images/train/image'
val_image_dir = 'DeepFashion2/deepfashion2_original_images/validation/image'
train_anno_dir = 'DeepFashion2/deepfashion2_original_images/train/annos'
val_anno_dir = 'DeepFashion2/deepfashion2_original_images/validation/annos'

custom_train_dir = 'custom_dataset/train/images'
custom_val_dir = 'custom_dataset/val/images'
custom_train_anno_dir = 'custom_dataset/train/labels'
custom_val_anno_dir = 'custom_dataset/val/labels'

os.makedirs(custom_train_dir, exist_ok=True)
os.makedirs(custom_val_dir, exist_ok=True)
os.makedirs(custom_train_anno_dir, exist_ok=True)
os.makedirs(custom_val_anno_dir, exist_ok=True)

train_images = [f for f in os.listdir(train_image_dir) if f.endswith('.jpg')]
val_images = [f for f in os.listdir(val_image_dir) if f.endswith('.jpg')]

random.seed(42)  
selected_train = random.sample(train_images, 40000)
selected_val = random.sample(val_images, 10000)

for img in selected_train:
    shutil.copy(os.path.join(train_image_dir, img), custom_train_dir)
    anno_file = img.replace('.jpg', '.json')
    shutil.copy(os.path.join(train_anno_dir, anno_file), custom_train_anno_dir)

for img in selected_val:
    shutil.copy(os.path.join(val_image_dir, img), custom_val_dir)
    anno_file = img.replace('.jpg', '.json')
    shutil.copy(os.path.join(val_anno_dir, anno_file), custom_val_anno_dir)