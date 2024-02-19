import csv
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
import argparse
import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

from models import GiMeFive

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = GiMeFive().to(device)
model.load_state_dict(torch.load('best_GiMeFive.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    # transforms.RandomHorizontalFlip(), 
    # transforms.RandomApply([
    #     transforms.RandomRotation(5),
    #     transforms.RandomCrop(64, padding=8)
    # ], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.RandomErasing(scale=(0.02,0.25)),
])

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores

def process_folder(folder_path):
    results = []
    for img_filename in os.listdir(folder_path):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_filename)
            scores = classify_image(img_path)
            results.append([img_path] + scores) 
    return results

def main(folder_path):
    results = process_folder(folder_path)
    header = ['filepath', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    with open('classification_scores_test.csv', 'w', newline='') as file: # change here!!!
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify images in a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images.')
    args = parser.parse_args()
    main(args.folder_path)
    
    # IMAGE_PATH = 'data/valid' # Please change this to your own path
    # # IMAGE_PATH = 'archive/RAF-DB/train'
    # # IMAGE_PATH = 'archive/RAF-DB/test'
    # main(IMAGE_PATH)