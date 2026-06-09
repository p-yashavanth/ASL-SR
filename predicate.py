import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import pyttsx3
from normalize_utils import normalize_landmarks

# Load model and labels
MODEL_PATH = "gesture_model.pkl"
LABELS_PATH = "labels.pkl"

try:
    model = joblib.load(MODEL_PATH)
    # Cast to plain Python strings to avoid np.str_ display issues
    labels = [str(l) for l in joblib.load(LABELS_PATH)]
    print("Model and labels loaded successfully.")
    print("Classes:", labels)
except Exception as e:
    print(f"Error loading model files: {e}")
    exit(1)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Text-to-Speech function running in a background thread to prevent GUI freezing
def speak_sentence(text):
    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as ex:
            print(f"TTS Error: {ex}")
    
    # Run in daemon thread
    t = threading.Thread(target=run_tts, daemon=True)
    t.start()

# Spelling / Debounce State
current_sentence = ""
last_letter = "?"
last_time = time.time()
added_1_5s = False
added_3_0s = False
cooldown_percent = 0.0

# Capture Webcam
cap = cv2.VideoCapture(0)

print("\n--- Key Controls ---")
print("SPACE     -> Add space to sentence")
print("BACKSPACE -> Delete last character")
print("ENTER     -> Speak sentence and clear")
print("C         -> Clear sentence")
print("Q         -> Quit application\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Flip horizontally for natural mirror display
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_pred = "?"
    confidence = 0.0
    color = (0, 0, 255)  # Default color (Red) for "?" or low confidence

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw skeleton connections
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 63 flat landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Normalize using the aspect-ratio corrected shared module
            aspect_ratio = w / h
            normalized = normalize_landmarks(landmarks, aspect_ratio=aspect_ratio)

            # Predict probabilities
            proba = model.predict_proba(normalized)[0]
            confidence = proba.max()
            pred_idx = proba.argmax()
            predicted_label = labels[pred_idx]

            # Only accept prediction if confidence > 0.75
            if confidence > 0.75:
                current_pred = str(predicted_label)
                # Color code based on confidence: Green > 90%, Yellow > 75%
                if confidence > 0.90:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 255, 255)  # Yellow
            else:
                current_pred = "?"
                color = (0, 0, 255)  # Red

    # Debounce / Letter Committing Logic
    current_time = time.time()
    if current_pred != "?":
        if current_pred == last_letter:
            elapsed = current_time - last_time

            # Update cooldown bar percent
            if not added_1_5s:
                cooldown_percent = min(elapsed / 1.5, 1.0)
            elif not added_3_0s:
                cooldown_percent = min((elapsed - 1.5) / 1.5, 1.0)
            else:
                cooldown_percent = 1.0

            # Commit second instance of letter (double letters like "LL") after 3.0s total
            if elapsed >= 3.0 and not added_3_0s:
                if current_pred == "space":
                    current_sentence += " "
                elif current_pred == "del":
                    current_sentence = current_sentence[:-1]
                else:
                    current_sentence += current_pred
                added_3_0s = True
            
            # Commit first instance of letter after 1.5s
            elif elapsed >= 1.5 and not added_1_5s:
                if current_pred == "space":
                    current_sentence += " "
                elif current_pred == "del":
                    current_sentence = current_sentence[:-1]
                else:
                    current_sentence += current_pred
                added_1_5s = True
        else:
            # Reset countdown for new letter
            last_letter = current_pred
            last_time = current_time
            added_1_5s = False
            added_3_0s = False
            cooldown_percent = 0.0
    else:
        # Reset state if no valid prediction is active
        last_letter = "?"
        last_time = current_time
        added_1_5s = False
        added_3_0s = False
        cooldown_percent = 0.0

    # Draw HUD Elements
    # 1. Detected Letter display (top-left)
    display_pred = current_pred
    if current_pred == "space":
        display_pred = "SPACE"
    elif current_pred == "del":
        display_pred = "DELETE"
        
    cv2.putText(frame, display_pred, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

    # 2. Confidence percentage
    if current_pred != "?":
        cv2.putText(frame, f"{confidence * 100:.0f}%", (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(frame, "---", (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)

    # 3. Cooldown progress bar (fills from left to right over 1.5s)
    # Background bar
    cv2.rectangle(frame, (50, 135), (250, 145), (60, 60, 60), -1)
    # Filled progress bar
    bar_width = int(cooldown_percent * 200)
    if bar_width > 0:
        cv2.rectangle(frame, (50, 135), (50 + bar_width, 145), color, -1)

    # 4. Subtitle bar at the bottom displaying current sentence
    # Draw semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw sentence text
    cv2.putText(frame, f"Sentence: {current_sentence}", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Render frame
    cv2.imshow("ASL Sign Recognition - Standalone", frame)

    # Key Listeners
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == 32:  # SPACE bar
        current_sentence += " "
        print("Space added.")
    elif key == 8:  # BACKSPACE key
        current_sentence = current_sentence[:-1]
        print("Backspace triggered.")
    elif key == 13:  # ENTER key
        if current_sentence.strip():
            print(f"Speaking sentence: {current_sentence}")
            speak_sentence(current_sentence)
        current_sentence = ""
    elif key == ord('c') or key == ord('C'):
        current_sentence = ""
        print("Sentence cleared.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
