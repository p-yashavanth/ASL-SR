# ASL Sign Recognition using Machine Learning
This project recognizes American Sign Language (ASL) alphabets in real time using a webcam.
Hand landmarks are extracted using MediaPipe, and a Random Forest classifier predicts the corresponding alphabet.
The system can be used for communication assistance and educational purposes.

Features:-
Real-time webcam recognition
MediaPipe hand tracking
Random Forest classifier
High prediction accuracy
Easy to train with new gestures
Lightweight model

project workflow:-
Load  Dataset
        ↓
MediaPipe detects hand landmarks
        ↓
Save landmarks
        ↓
Train Random Forest
        ↓
Save Model
        ↓
Real-time Prediction
