import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from models import GiMeFive
import numpy as np
import cv2


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class_labels = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

model = GiMeFive().to(device)
model.load_state_dict(torch.load('best_GiMeFive.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# access webcam as numpy.ndarray
video_capture = cv2.VideoCapture(0)

# text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # BGR color
thickness = 2
line_type = cv2.LINE_AA

max_emotion = ''

def detect_emotion(video_frame):
    vid_fr_tensor = transform(video_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(vid_fr_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    # print(f'rounded_scores in detect_emotion {rounded_scores}')
    return rounded_scores

def get_max_emotion(x, y, w, h, video_frame):
    crop_img = video_frame[y : y + h, x : x + w]
    pil_crop_img = Image.fromarray(crop_img)
    # slower cropping
    rounded_scores = detect_emotion(pil_crop_img)    
    # get index from max value in rounded_scores
    max_index = np.argmax(rounded_scores)
    max_emotion = class_labels[max_index]
    # print(f'max_emotion: {max_emotion}')

    return max_emotion

def print_max_emotion(x, y, video_frame, max_emotion):
    # position to put the text for the max emotion
    org = (x, y - 15)
    cv2.putText(video_frame, max_emotion, org, font, font_scale, font_color, thickness, line_type)
    
def print_all_emotion(x, y, w, h, video_frame):
    crop_img = video_frame[y : y + h, x : x + w]
    pil_crop_img = Image.fromarray(crop_img)
    # slower cropping
    rounded_scores = detect_emotion(pil_crop_img)
    # print(f'rounded_scores in detect_bounding_box: {rounded_scores}')
    # create text to be displayed
    org = (x + w + 10, y - 20)
    for index, value in enumerate(class_labels):
        emotion_str = (f'{value}: {rounded_scores[index]:.2f}')
        y = org[1] + 40
        org = (org[0], y)
        cv2.putText(video_frame, emotion_str, org, font, font_scale, font_color, thickness, line_type)
    
# identify Face in Video Stream
def detect_bounding_box(video_frame, counter):
    global max_emotion
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        # draw bounding box on face
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # crop bounding box
        if counter == 0:
            max_emotion = get_max_emotion(x, y, w, h, video_frame) 
        
        print_max_emotion(x, y, video_frame, max_emotion) # displays the max_emotion according to evaluation_frequency
        print_all_emotion(x, y, w, h, video_frame) 
        # evaluates every video_frame for debugging

    return faces

counter = 0
evaluation_frequency = 5

# Loop for Real-Time Face Detection
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    
    faces = detect_bounding_box(video_frame, counter)  # apply the function we created to the video frame, faces as variable not used
    
    cv2.imshow("My Face Detection Project", video_frame)  # display the processed frame in a window named "My Face Detection Project"

    # print(type(video_frame))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    counter += 1
    if counter == evaluation_frequency:
        counter = 0
        
video_capture.release()
cv2.destroyAllWindows()