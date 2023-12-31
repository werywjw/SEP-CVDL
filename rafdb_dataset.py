import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class RAFDBDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label
    
if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])
    
    rafdb_dataset_train = RAFDBDataset(csv_file='archive/train_labels.csv',
                             img_dir='archive/DATASET/train/',
                             transform=transform)
    
    data_train_loader = DataLoader(rafdb_dataset_train, batch_size=64, shuffle=True, num_workers=4)

    train_image, train_label = next(iter(data_train_loader))
    print(f"Train batch: image shape {train_image.shape}, labels shape {train_label.shape}")
