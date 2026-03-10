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
DEBUG_CONSOLE = True
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

def recognize_point_to_screen(landmarks: list[tuple]) -> bool:
    index_tip = landmarks[8]
    # Temporary (to be changed with implementation)
    return False

def pinky_is_straight(landmarks: list[tuple]) -> bool:
    pinky_pip = landmarks[18]
    pinky_dip = landmarks[19]
    pinky_tip = landmarks[20]

    # We have some tolerance, but generally expect minimal distance
    # between each adjacent marked point on the pinky finger
    pinky_pip_dip_distance = distance(pinky_pip, pinky_dip)
    pinky_dip_tip_distance = distance(pinky_dip, pinky_tip)
    if DEBUG_CONSOLE:
        print(f'Pinky: Pip to Dip distance == {pinky_pip_dip_distance}')
        print(f'Pinky: Dip to Tip distance == {pinky_dip_tip_distance}')
    
    return pinky_pip_dip_distance < 0.05 and pinky_dip_tip_distance < 0.05

# We could just check that the pinky is close to the third finger tip,
# but we need to distinguish between which hand by using coordinate
# directions too
def recognize_left_hand_pointing_right(landmarks: list[tuple]) -> bool:
    wrist = landmarks[0]
    pinky_mcp = landmarks[17]
    pinky_tip = landmarks[20]
    third_finger_tip = landmarks[16]

    # If the tips of your third finger and pinky finger are close together,
    # your pinky is in a (relatively) straight line, and your pinky is to the
    # left of your third finger, then this is true
    pinky_third_tips_distance = distance(pinky_tip, third_finger_tip)
    if DEBUG_CONSOLE:
        print(f'LEFT hand pointing RIGHT test')
        print(f'pinky third tips distance: {pinky_third_tips_distance}')
        print(f'wrist.x == {wrist[0]} and pinky_mcp.x == {pinky_mcp[0]} - {wrist[0] < pinky_mcp[0]}')

    return pinky_third_tips_distance < 1 and wrist[0] < pinky_mcp[0] and pinky_is_straight(landmarks)

def recognize_right_hand_pointing_left(landmarks: list[tuple]) -> bool:
    wrist = landmarks[0]
    pinky_mcp = landmarks[17]
    pinky_tip = landmarks[20]
    third_finger_tip = landmarks[16]

    # If the tips of your third finger and pinky finger are close together,
    # your pinky is in a (relatively) straight line, and your pinky is to the
    # left of your third finger, then this is true
    pinky_third_tips_distance = distance(pinky_tip, third_finger_tip)
    if DEBUG_CONSOLE:
        print(f'RIGHT hand pointing LEFT test')
        print(f'pinky third tips distance: {pinky_third_tips_distance}')
        print(f'wrist.x == {wrist[0]} and pinky_mcp.x == {pinky_mcp[0]} - {wrist[0] > pinky_mcp[0]}')

    return pinky_third_tips_distance < 1 and wrist[0] > pinky_mcp[0] and pinky_is_straight(landmarks)

SCROLL_LENGTH = 100


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally and convert the BGR image to RGB.
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        timestamp = int(time.time() * 1000)

        # Run both recognition models
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
        gesture_result = gesture_recognizer.recognize_for_video(mp_image, timestamp)

        detected_text = "None"

        # Draw the gesture recognition results on the image
        if gesture_result.gestures:
            recognized_gesture = gesture_result.gestures[0][0].category_name
            confidence = gesture_result.gestures[0][0].score

            # Pressing keys for discord shortcuts with pyautogui based on recognized gesture
            if confidence > 0.51:
                if recognized_gesture == "Thumb_Up":
                    pyautogui.scroll(SCROLL_LENGTH)
                    detected_text = f"CANNED: {recognized_gesture}"
                elif recognized_gesture == "Thumb_Down":
                    pyautogui.scroll(-SCROLL_LENGTH)
                    detected_text = f"CANNED: {recognized_gesture}"

        if hand_result.hand_landmarks:
            lm = hand_result.hand_landmarks[0]
            landmarks = [(p.x, p.y) for p in lm]
            # Draw landmarks
            for x, y in landmarks:
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
            if detected_text == "None":
                if recognize_point_to_screen(landmarks):
                    pyautogui.leftClick()
                elif recognize_left_hand_pointing_right(landmarks):
                    # Go to Next Server in Discord
                    pyautogui.hotkey('ctrl', 'alt', 'down')
                elif recognize_right_hand_pointing_left(landmarks):
                    # Go to Previous Server in Discord
                    pyautogui.hotkey('ctrl', 'alt', 'up')

            # Display recognized gesture and confidence 
            cv2.putText(frame, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image (can comment this out for better performance later on)
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
