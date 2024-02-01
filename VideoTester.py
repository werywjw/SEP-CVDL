import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np

# Define the PyTorch model
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        # Define your model layers here

    def forward(self, x):
        # Define the forward pass
        return x

# Load the pre-trained PyTorch model
model = EmotionModel()
model.load_state_dict(torch.load('fer_pytorch.pth')) # placeholder for the filename of the PyTorch model weights file
model.eval()

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Preprocess the image for PyTorch model
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(roi_gray).unsqueeze(0)
        img_tensor = Variable(img_tensor).float()

        # Make a prediction
        predictions = model(img_tensor)

        # Find max index
        max_index = torch.argmax(predictions, dim=1).item()

        emotions = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()