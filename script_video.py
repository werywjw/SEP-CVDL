import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import dlib
import argparse
import textwrap

from models import GiMeFive
from hook import Hook

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

class_labels = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

model = GiMeFive().to(device)
model.load_state_dict(torch.load('best_GiMeFive.pth', map_location=device))
model.eval()

final_layer = model.conv5
hook = Hook()
hook.register_hook(final_layer)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_array = np.array(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    
    return rounded_scores, image, image_array, image_tensor


from pathlib import Path
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (154, 1, 254) # BGR color neon pink 254,1,154
thickness = 2
line_type = cv2.LINE_AA

max_emotion = ''
transparency = 0.4

def detect_emotion(pil_crop_img):
    # Convert NumPy array to PIL Image
    pil_crop_img = Image.fromarray(pil_crop_img)
    
    vid_fr_tensor = transform(pil_crop_img).unsqueeze(0).to(device)
    # with torch.no_grad():
    logits = model(vid_fr_tensor)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    predicted_class_idx = predicted_class.item()

    one_hot_output = torch.FloatTensor(1, probabilities.shape[1]).zero_()
    one_hot_output[0][predicted_class_idx] = 1
    logits.backward(one_hot_output, retain_graph=True)

    gradients = hook.backward_out
    feature_maps = hook.forward_out

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
    cam = cam.clamp(min=0).squeeze() 

    cam -= cam.min()
    cam /= cam.max()
    cam = cam.cpu().detach().numpy()

    # scores = probabilities.cpu().numpy().flatten()
    scores = probabilities.cpu().detach().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores, cam

def plot_heatmap(x, y, w, h, cam, pil_crop_img, video_frame):
    # resize cam to w, h
    cam = cv2.resize(cam, (w, h))
    
    # apply color map to resized cam
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Get the region of interest on the video frame
    roi = video_frame[y:y+h, x:x+w, :]

    # Blend the heatmap with the ROI
    overlay = heatmap * transparency + roi / 255 * (1 - transparency)
    overlay = np.clip(overlay, 0, 1)

    # Replace the ROI with the blended overlay
    video_frame[y:y+h, x:x+w, :] = np.uint8(255 * overlay)
        
def update_max_emotion(rounded_scores):  
    # get index from max value in rounded_scores
    max_index = np.argmax(rounded_scores)
    max_emotion = class_labels[max_index]
    return max_emotion # returns max_emotion as string

def print_max_emotion(x, y, max_emotion, video_frame):
    # position to put the text for the max emotion
    org = (x, y - 15)
    cv2.putText(video_frame, max_emotion, org, font, font_scale, font_color, thickness, line_type)
    
def print_all_emotion(x, y, w, rounded_scores, video_frame):
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
    
    # notes: MultiScale optimized
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(64, 64))

    for (x, y, w, h) in faces:
        cv2.rectangle(gray_image, (x, y), (x+w, y+h), (255, 0, 0), 0)

        # convert the ROI to a dlib rectangle
        dlib_rect = dlib.rectangle(x, y, x+w, y+h)

        # detect facial landmarks through dlib
        landmarks = predictor(gray_image, dlib_rect)

        pil_crop_img = video_frame[y : y + h, x : x + w]
        rounded_scores, cam = detect_emotion(pil_crop_img)
            
        if counter == 0:
            max_emotion = update_max_emotion(rounded_scores) 
            
        # draw landmarks on the video_frame
        for i in range(68):  # Assuming you have 68 landmarks
            cv2.circle(video_frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (255, 255, 255), 0)
            
        plot_heatmap(x, y, w, h, cam, pil_crop_img, video_frame)
        print_max_emotion(x, y, max_emotion, video_frame) # displays the max_emotion according to evaluation_frequency
        print_all_emotion(x, y, w, rounded_scores, video_frame) # evaluates every video_frame for debugging

    return faces

def create_video_out(source, input_path_to_video):
    if source == 'camera':
        video_capture = cv2.VideoCapture(0)
        fps = 10
        out_file_name = 'cam_eval_video.mp4'
    elif source == 'video':
        video_capture = cv2.VideoCapture(input_path_to_video)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        out_file_name = 'eval_video.mp4'
    else:
        print('unknown input')
        print('please enter camera or video')
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_file_name, fourcc, fps, (frame_width, frame_height))
    return out, video_capture


# loop for Real-Time Face Detection
def evaluate_input(source, input_path_to_video):
    out, video_capture = create_video_out(source, input_path_to_video)
    
    counter = 0
    evaluation_frequency = 5

    while True:

        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully
        
        faces = detect_bounding_box(video_frame, counter)  # apply the function we created to the video frame, faces as variable not used
        
        cv2.imshow("My Face Detection Project", video_frame)  # display the processed frame in a window named "My Face Detection Project"

        out.write(video_frame)  # write the processed frame to the output video file
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        counter += 1
        if counter == evaluation_frequency:
            counter = 0

    hook.unregister_hook()        
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


def main(args):
    print(args)
    evaluate_input(args.source, args.input_path_to_video)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            '''\
            This program performs emotion evaluation of faces from given video or camera feed.
            
            If "camera" is selected it will generate a video with analyzed emotions from live stream,
            and store it in the file cam_eval_video.mp4.
            
            If "video" is selected it will analyze emotions from given video,
            and store it in the file eval_video.mp4.
            ''')
    )
    parser.add_argument('-s', '--source', type=str, help='Enter "camera" or "video"', default='video')
    parser.add_argument('-i', '--input_path_to_video', type=str, help='Path to the video file.', default='video/test_video_noemotions.mp4')
    args = parser.parse_args()

    main(args)
