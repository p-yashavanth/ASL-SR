from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import pyttsx3
import os
import zipfile



from normalize_utils import normalize_landmarks

app = Flask(__name__)

# Load model and labels
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pkl")
ZIP_PATH = os.path.join(BASE_DIR, "gesture_model.zip")

try:
    if not os.path.exists(MODEL_PATH):
        print("Extracting gesture_model.zip...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)

    model = joblib.load(MODEL_PATH)
    labels = [str(l) for l in joblib.load(LABELS_PATH)]
except Exception as e:
    print(f"Error loading model files: {e}")
    exit(1)

# Global thread-safe spelling state
state = {
    "letter": "?",
    "confidence": 0.0,
    "sentence": "",
    "is_stable": False,
    "cooldown_percent": 0.0
}
state_lock = threading.Lock()

# Thread-safe text-to-speech engine wrapper
def speak_sentence(text):
    def run_tts():
        try:
            # Initialize inside the thread to avoid COM apartment thread issues on Windows
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as ex:
            print(f"TTS Thread Error: {ex}")
            
    t = threading.Thread(target=run_tts, daemon=True)
    t.start()

# MJPEG Stream generator
def generate_frames():
    global state
    
    # Initialize Camera and MediaPipe Hands in the stream thread
    cap = cv2.VideoCapture(0)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    # Debounce tracking variables
    last_letter = "?"
    last_time = time.time()
    last_match_time = time.time()
    
    print("Webcam stream started.")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                # Tiny sleep to avoid busy-waiting if webcam drops
                time.sleep(0.03)
                continue
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # MediaPipe hands processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            current_pred = "?"
            confidence = 0.0
            is_stable = False
            cooldown_percent = 0.0
            color = (0, 0, 255) # Red for "?"
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw skeletal outline on frame
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract 63 coordinates
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                        
                    # Normalize using the aspect-ratio corrected shared module
                    aspect_ratio = w / h
                    normalized = normalize_landmarks(landmarks, aspect_ratio=aspect_ratio)
                    
                    # Predict class probabilities
                    proba = model.predict_proba(normalized)[0]
                    confidence = proba.max()
                    pred_idx = proba.argmax()
                    predicted_label = labels[pred_idx]
                    
                    if confidence > 0.75:
                        current_pred = str(predicted_label)
                        if confidence > 0.90:
                            color = (0, 255, 0) # Green
                        else:
                            color = (0, 255, 255) # Yellow
                    else:
                        current_pred = "?"
                        color = (0, 0, 255)
            
            # Jitter-resistant Debounce Logic (1.5s hold + 0.15s tolerance)
            current_time = time.time()
            if current_pred != "?":
                if current_pred == last_letter:
                    # Match: update match timestamp and calculate elapsed time
                    last_match_time = current_time
                    elapsed = current_time - last_time
                    
                    cooldown_percent = min(elapsed / 1.5, 1.0)
                    is_stable = (elapsed >= 1.5)
                    
                    if elapsed >= 1.5:
                        with state_lock:
                            if current_pred == "space":
                                state["sentence"] += " "
                            elif current_pred == "del":
                                state["sentence"] = state["sentence"][:-1]
                            else:
                                state["sentence"] += current_pred
                        # Reset hold timer to support auto-repeat commits
                        last_time = current_time
                        cooldown_percent = 0.0
                else:
                    # Mismatch: check if within the 0.15s jitter tolerance
                    if current_time - last_match_time > 0.15:
                        # Real transition: reset state to this new letter
                        last_letter = current_pred
                        last_time = current_time
                        last_match_time = current_time
                        cooldown_percent = 0.0
                        is_stable = False
                    else:
                        # Ignored glitch: keep counting the previous stable hold
                        elapsed = current_time - last_time
                        cooldown_percent = min(elapsed / 1.5, 1.0)
                        is_stable = (elapsed >= 1.5)
            else:
                # Predict is "?"
                if current_time - last_match_time > 0.15:
                    last_letter = "?"
                    last_time = current_time
                    last_match_time = current_time
                    cooldown_percent = 0.0
                    is_stable = False
                else:
                    # Ignored glitch
                    elapsed = time.time() - last_time
                    cooldown_percent = min(elapsed / 1.5, 1.0)
                    is_stable = (elapsed >= 1.5)
                
            # Update global state values
            with state_lock:
                state["letter"] = current_pred
                state["confidence"] = float(confidence)
                state["is_stable"] = is_stable
                state["cooldown_percent"] = float(cooldown_percent)
                
            # Draw clean overlay HUD directly on the MJPEG stream
            display_pred = current_pred
            if current_pred == "space":
                display_pred = "SPACE"
            elif current_pred == "del":
                display_pred = "DELETE"
                
            cv2.putText(frame, display_pred, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            if current_pred != "?":
                cv2.putText(frame, f"{confidence*100:.0f}%", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            else:
                cv2.putText(frame, "---", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 2)
                
            # Draw thin HUD cooldown indicator bar on video frame
            cv2.rectangle(frame, (30, 95), (150, 102), (60, 60, 60), -1)
            bar_w = int(cooldown_percent * 120)
            if bar_w > 0:
                cv2.rectangle(frame, (30, 95), (30 + bar_w, 102), color, -1)
                
            # Compress frame and yield as MJPEG boundary block
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    finally:
        # Safeguard camera release when route thread is terminated or browser disconnects
        cap.release()
        hands.close()
        print("Webcam stream released.")

# Server Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return MJPEG multipart stream response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    global state
    with state_lock:
        # Return a copy of the state
        return jsonify({
            "letter": state["letter"],
            "confidence": state["confidence"],
            "sentence": state["sentence"],
            "is_stable": state["is_stable"],
            "cooldown_percent": state["cooldown_percent"]
        })

@app.route('/action', methods=['POST'])
def action():
    global state
    data = request.get_json()
    act = data.get("action")
    
    with state_lock:
        if act == "space":
            state["sentence"] += " "
            print("Action: Space added via web button.")
        elif act == "backspace":
            state["sentence"] = state["sentence"][:-1]
            print("Action: Backspace triggered via web button.")
        elif act == "clear":
            state["sentence"] = ""
            print("Action: Sentence cleared via web button.")
        elif act == "speak":
            sentence_to_speak = state["sentence"]
            if sentence_to_speak.strip():
                print(f"Action: Speaking sentence: '{sentence_to_speak}'")
                speak_sentence(sentence_to_speak)
                
        return jsonify({"status": "success", "sentence": state["sentence"]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
