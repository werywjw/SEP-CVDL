import os
import pandas as pd

path = 'dataset/validation_set'

label_mapping = {
    "happiness": 0,
    "surprise": 1,
    "sadness": 2,
    "anger": 3,
    "disgust": 4,
    "fear": 5,
}

image_data = []

for filename in os.listdir(path):
    if filename.endswith(".jpg"):  
        label_name = filename.split('_')[1].split('.')[0]
        label_value = label_mapping.get(label_name)
        if label_value is not None:  
            image_data.append([filename, label_value])

df = pd.DataFrame(image_data, columns=["ImageName", "Label"])

csv_file_path = 'dataset/vali_labels.csv'

df.to_csv(csv_file_path, index=False, header=False)

print(f"CSV file created at: {csv_file_path}")