from flask import Flask, render_template, Response, stream_with_context
from keras.models import load_model
import cv2
import mediapipe as mp
import os
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Initialize MediaPipe hand detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Loading the pre-trained MediaPipe hand detection model
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.3)

# To detect and return the hand bounding box


def detect_hand(hand_landmarks, frame):
    height, width, _ = frame.shape
    landmarks_x = [landmark.x * width for landmark in hand_landmarks.landmark]
    landmarks_y = [landmark.y * height for landmark in hand_landmarks.landmark]
    x_min = int(min(landmarks_x))-20
    x_max = int(max(landmarks_x))+20
    y_min = int(min(landmarks_y))-20
    y_max = int(max(landmarks_y))+20

    if x_min != -1 and y_min != -1:
        hand_width = x_max - x_min
        hand_height = y_max - y_min

        # Calculate the center of the hand bounding box
        center_x = x_min + hand_width // 2
        center_y = y_min + hand_height // 2

        # Calculate the starting and ending coordinates for the 224x224 region
        crop_size = 224
        crop_start_x = center_x - crop_size // 2
        crop_start_y = center_y - crop_size // 2
        crop_end_x = crop_start_x + crop_size
        crop_end_y = crop_start_y + crop_size

        # Ensure the cropping region is within the frame boundaries
        crop_start_x = max(crop_start_x, 0)
        crop_start_y = max(crop_start_y, 0)
        crop_end_x = min(crop_end_x, width)
        crop_end_y = min(crop_end_y, height)

    # Check if the crop coordinates are within the frame boundaries
    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        return -1, -1, -1, -1  # Invalid crop coordinates

    return crop_start_x, crop_start_y, crop_end_x, crop_end_y


# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the pre-trained model
model = load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            ret, frame = cap.read()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform hand detection
            results = hands.process(frame_rgb)

            # If hand is detected, crop and save the image
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Get bounding box coordinates of the hand
                x_min, y_min, x_max, y_max = detect_hand(hand_landmarks, frame)

                # If the bounding box is valid, crop and save the image
                if x_min != -1 and y_min != -1:
                    cropped_image = frame[y_min:y_max, x_min:x_max]
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())
                    img1 = cv2.resize(cropped_image, (224, 224),
                                      interpolation=cv2.INTER_AREA)
                    img1 = image.img_to_array(img1)
                    img1 = np.expand_dims(img1, axis=0)
                    pred = np.argmax(model.predict(img1))

                    output = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                              'L', 'O', 'P', 'Q', 'R', 'S', 'U', 'V', 'W', 'X', 'Y']

                    cv2.putText(frame, str(
                        output[pred]), (x_min, y_min-10), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    if x_min != -1 and y_min != -1:
                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_max, y_max), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
