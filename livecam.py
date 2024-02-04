import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from model import EmotionClassifier
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %%
# !! {"metadata":# !! {}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")

class_labels = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

model = EmotionClassifier().to(device)
model.load_state_dict(torch.load('best_RAF.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# %%
# !! {"metadata":# !! {}
# Grad-CAM with Torchcam: https://github.com/frgfm/torch-cam/blob/main/README.md

# %%
# !! {"metadata":# !! {}
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# access webcam as numpy.ndarray
video_capture = cv2.VideoCapture(0)

def detect_emotion(video_frame):
    vid_fr_tensor = transform(video_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(vid_fr_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    # print(f'rounded_scores in detect_emotion {rounded_scores}')
    return rounded_scores

# identify Face in Video Stream
def detect_bounding_box(video_frame):
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        # draw bounding box on face
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # crop bounding box
        crop_img = video_frame[y : y + h, x : x + w]
        pil_crop_img = Image.fromarray(crop_img)
        rounded_scores = detect_emotion(pil_crop_img)
        # print(f'rounded_scores in detect_bounding_box: {rounded_scores}')
        
        # create text to be displayed
        emotion_evaluation_str = []
        for index, value in enumerate(class_labels):
            emotion_evaluation_str.append(f'{value}: {rounded_scores[index]:.2f}')
            
        # get index from max value in rounded_scores
        max_index = np.argmax(rounded_scores)
        max_emotion = class_labels[max_index]

        # text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # BGR color
        thickness = 2
        line_type = cv2.LINE_AA
        # line_height = 40
        
        # position to put the text for the max emotion
        org = (x, y - 15)
        cv2.putText(video_frame, max_emotion, org, font, font_scale, font_color, thickness, line_type)
        
        # position to put the text for 6 emotions
        # org = (x + w + 10, y + 20)
        # cv2.putText(video_frame, max_emotion, org, font, font_scale, font_color, thickness, line_type)
        
        # # put each line of text on the image
        # for i, line in enumerate(emotion_evaluation_str):
        #     # Calculate the position for this line
        #     y = org[1] + i * line_height

    return faces

# Loop for Real-Time Face Detection
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    # print(type(video_frame))
    
    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
