import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions, vision

import pyautogui
import numpy as np
import time

# Path to the gesture recognition model
GESTURE_MODEL = "gesture_recognizer.task"  # Update this to the correct path where the model is saved, if not in current directory
HAND_MODEL = 'hand_landmarker.task'

# Initialize the Gesture Recognizer

VisionRunningMode = vision.RunningMode

gesture_options = vision.GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

gesture_recognizer = vision.GestureRecognizer.create_from_options(
    gesture_options
)

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Custom gestures we define with the hand_landmarker
def recognize_ok(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return distance(thumb_tip, index_tip) < 0.05

def recognize_left_hand_pointing_right(landmarks):
    pass

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally and convert the BGR image to RGB.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp = int(time.time() * 1000)

        # Perform gesture recognition on the image
        result = gesture_recognizer.recognize_for_video(mp_image, timestamp)
        # Also run the hand_landmarker model
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)

        # Draw the gesture recognition results on the image
        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score

            # Pressing keys for discord shortcuts with pyautogui based on recognized gesture
            if recognized_gesture == "Thumb_Up":
                pyautogui.scroll(20)
            elif recognized_gesture == "Thumb_Down":
                pyautogui.scroll(-20)
            elif recognized_gesture == "Pointing_Up":
                pyautogui.leftClick()
                time.sleep(1)

        if hand_result.hand_landmarks:
            lm = hand_result.hand_landmarks[0]
            landmarks = [(p.x, p.y) for p in lm]
            if recognize_ok(landmarks):
                pyautogui.hotkey('ctrl', 'alt', 'down')
                time.sleep(1)

            # Display recognized gesture and confidence 
            cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image (can comment this out for better performance later on)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
