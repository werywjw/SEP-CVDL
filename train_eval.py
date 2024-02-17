import torch
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.model_selection import ParameterGrid

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
# print(f"Using device: {device}")

from get_dataset import GiMeFiveDataset
from model import GiMeFive
from model import SEBlock, ResidualBlock, GiMeFiveRes, VGG16, BasicBlock, ResNet, EmotionClassifierResNet18

def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(scale=(0.02,0.25)),
    ])
        
    # rafdb_dataset_train = GiMeFiveDataset(csv_file='archive/RAF-DB/train_RAF_labels.csv',
    #                             img_dir='archive/RAF-DB/train/',
    #                             transform=transform)

    rafdb_dataset_train = GiMeFiveDataset(csv_file='archive/FER2013/train_FER_labels.csv',
                                img_dir='archive/FER2013/train/',
                                transform=transform)

    # rafdb_dataset_train = GiMeFiveDataset(csv_file='data/train_labels.csv',
    #                             img_dir='data/train/',
    #                             transform=transform)
    data_train_loader = DataLoader(rafdb_dataset_train, batch_size=16, shuffle=True, num_workers=4)
    train_image, train_label = next(iter(data_train_loader))
    print(f"Train batch: image shape {train_image.shape}, labels shape {train_label.shape}")

    rafdb_dataset_vali = GiMeFiveDataset(csv_file='data/valid_labels.csv',
                                img_dir='data/valid',
                                transform=transform)
    data_vali_loader = DataLoader(rafdb_dataset_vali, batch_size=16, shuffle=False, num_workers=0)
    vali_image, vali_label = next(iter(data_vali_loader))
    print(f"Vali batch: image shape {vali_image.shape}, labels shape {vali_label.shape}")

    # rafdb_dataset_test = GiMeFiveDataset(csv_file='archive/RAF-DB/test_RAF_labels.csv',
    #                             img_dir='archive/RAF-DB/test/',
    #                             transform=transform)

    rafdb_dataset_test = GiMeFiveDataset(csv_file='archive/FER2013/test_FER_labels.csv',
                                img_dir='archive/FER2013/test/',
                                transform=transform)

    # rafdb_dataset_test = GiMeFiveDataset(csv_file='data/test_labels.csv',
    #                             img_dir='data/test/',
    #                             transform=transform)
    data_test_loader = DataLoader(rafdb_dataset_test, batch_size=16, shuffle=False, num_workers=0)
    test_image, test_label = next(iter(data_test_loader))
    print(f"Test batch: image shape {test_image.shape}, labels shape {test_label.shape}")


    model = GiMeFive().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    patience = 15
    best_val_acc = 0  
    patience_counter = 0

    num_epochs = 80

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(data_train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(data_train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in data_test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss = test_running_loss / len(data_test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in data_vali_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(data_vali_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epochs.")
        
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

    # Plotting and saving results

    # plt.figure(figsize=(15, 7))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, 80), train_losses, label='Train Loss') # change this number after '(1, _)' to num_epochs+1
    # plt.plot(range(1, 80), test_losses, label='Test Loss') # change this number after '(1, _)' to num_epochs+1
    # plt.plot(range(1, 80), val_losses, label='Validation Loss') # change this number after '(1, _)' to num_epochs+1
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Losses on GiMeFive') # change
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, 80), train_accuracies, label='Train Accuracy') # change this number after '(1, _)' to num_epochs+1
    # plt.plot(range(1, 80), test_accuracies, label='Test Accuracy') # change this number after '(1, _)' to num_epochs+1
    # plt.plot(range(1, 80), val_accuracies, label='Validation Accuracy') # change this number after '(1, _)' to num_epochs+1
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracies on GiMeFive') # change
    # plt.legend()

    # plt.show()


    # df = pd.DataFrame({
    #     'Epoch': range(1, 60), # change this number after '(1, _)' to num_epochs+1
    #     'Train Loss': train_losses,
    #     'Test Loss': test_losses,
    #     'Validation Loss': val_losses,
    #     'Train Accuracy': train_accuracies,
    #     'Test Accuracy': test_accuracies,
    #     'Validation Accuracy': val_accuracies
    # })
    # df.to_csv('result_gimefive.csv', index=False) # change this CSV


if __name__ == '__main__':
    main()