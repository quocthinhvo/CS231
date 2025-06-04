import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image
import os
import argparse

NUM_CLASSES = 13
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    image = test_transforms(image)
    image = image.unsqueeze(0) 
    image = image.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    
    return predicted_class

def predict_folder(model, folder_path, class_names):
    predictions = []
    for img_name in os.listdir(folder_path):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_name)
            predicted_class = predict_image(model, img_path, class_names)
            predictions.append((img_name, predicted_class))
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Predict using trained CNN model")
    parser.add_argument('--model_path', type=str, default="best_model.pth", help="Path to trained model")
    parser.add_argument('--input', type=str, required=True, help="Path to image or folder of images")
    parser.add_argument('--class_names', type=str, nargs='+', default=None, help="List of class names")
    
    args = parser.parse_args()
    
    if args.class_names is None:
        train_dir = "train" 
        args.class_names = sorted(os.listdir(train_dir))
    
    model = load_model(args.model_path)
    
    if os.path.isfile(args.input):
        predicted_class = predict_image(model, args.input, args.class_names)
        print(f"Image: {args.input} -> Predicted class: {predicted_class}")
    elif os.path.isdir(args.input):
        predictions = predict_folder(model, args.input, args.class_names)
        for img_name, predicted_class in predictions:
            print(f"Image: {img_name} -> Predicted class: {predicted_class}")
    else:
        print("Error: Input must be a valid image file or folder path.")

if __name__ == "__main__":
    main()