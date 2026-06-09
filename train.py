"""
train.py - Robust ASL Model Trainer (strictly CSV + Augmentation)
================================================================
Trains a gesture recognition model on ground-truth MediaPipe landmark coordinates.
Applies extensive offline data augmentation to ensure high accuracy and stability:
- Mirroring (left/right hand support)
- Rotation (tilt resilience)
- Scaling (distance/hand size resilience)
- Gaussian noise (sensor jitter resilience)
"""

import os
import csv
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from normalize_utils import normalize_landmarks

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH    = r"C:\Users\P Yashvanth\Downloads\asl_landmarks_final.csv"
MODEL_PATH  = "gesture_model.pkl"
LABELS_PATH = "labels.pkl"

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════════
def augment_landmarks(coords_21x3):
    """
    Applies mirror flipping, rotations, scaling, and noise to raw landmarks (21x3).
    Returns a list of augmented numpy arrays.
    """
    augmented = []
    
    # 1. Base sample
    augmented.append(coords_21x3)
    
    # 2. Flip (mirror x axis)
    flipped = coords_21x3.copy()
    flipped[:, 0] = -flipped[:, 0]
    augmented.append(flipped)
    
    # Center around wrist (landmark 0) for rotation and scaling
    wrist = coords_21x3[0]
    centered = coords_21x3 - wrist
    
    # 3. Rotations in XY plane (angles: -20, -10, 10, 20 degrees)
    for angle_deg in [-20, -10, 10, 20]:
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        rotated = centered.dot(rot_matrix.T) + wrist
        augmented.append(rotated)
        
        # Also flip the rotated versions
        rotated_flipped = rotated.copy()
        rotated_flipped[:, 0] = -rotated_flipped[:, 0]
        augmented.append(rotated_flipped)
        
    # 4. Scaling (factors: 0.9, 0.95, 1.05, 1.1)
    for scale in [0.9, 0.95, 1.05, 1.1]:
        scaled = centered * scale + wrist
        augmented.append(scaled)
        
        # Also flip the scaled versions
        scaled_flipped = scaled.copy()
        scaled_flipped[:, 0] = -scaled_flipped[:, 0]
        augmented.append(scaled_flipped)
        
    # 5. Gaussian noise
    for _ in range(2):
        # Apply small jitter (std dev = 0.005)
        noisy = coords_21x3 + np.random.normal(0, 0.005, coords_21x3.shape)
        augmented.append(noisy)
        
        # Also flip the noisy versions
        noisy_flipped = noisy.copy()
        noisy_flipped[:, 0] = -noisy_flipped[:, 0]
        augmented.append(noisy_flipped)
        
    return augmented


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_augment_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV dataset not found at {CSV_PATH}")
        return [], []

    X, y = [], []
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        label_idx = header.index("label")
        
        print(f"Reading dataset from {CSV_PATH}...")
        for row in reader:
            if not row:
                continue
            try:
                # Extract 63 flat coordinates
                coords = [float(v) for v in row[:63]]
                label  = row[label_idx].strip()
                
                # Reshape to (21, 3)
                coords_arr = np.array(coords, dtype=np.float32).reshape(21, 3)
                
                # Generate augmented samples
                aug_samples = augment_landmarks(coords_arr)
                
                for sample in aug_samples:
                    # Normalize using aspect_ratio=1.3333 (matches CSV recording frame shape)
                    norm = normalize_landmarks(sample.flatten(), aspect_ratio=1.3333).flatten()
                    X.append(norm)
                    y.append(label)
            except Exception as e:
                pass

    print(f"Loaded {len(X)//22} original samples -> Augmented to {len(X)} total samples.")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("ASL Model Trainer — Augmented CSV Pipeline")
    print("=" * 60)
    
    X, y = load_and_augment_dataset()
    if len(X) == 0:
        print("No training data loaded. Aborting.")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    # Train / test split (80/20 stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_tr)}  |  Test samples: {len(X_te)}")
    print(f"Classes ({len(set(y))}): {sorted(set(y))}")
    
    # ── Model 1: Random Forest ─────────────────────────────────────────────────
    print("\nTraining Random Forest (300 trees)...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te))
    print(f"  RF test accuracy : {rf_acc*100:.2f}%")
    
    # ── Model 2: Extra Trees ───────────────────────────────────────────────────
    print("\nTraining Extra Trees (300 trees)...")
    et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    et.fit(X_tr, y_tr)
    et_acc = accuracy_score(y_te, et.predict(X_te))
    print(f"  ET test accuracy : {et_acc*100:.2f}%")
    
    # ── Select winner ──────────────────────────────────────────────────────────
    if et_acc >= rf_acc:
        best, best_acc, name = et, et_acc, "ExtraTrees"
    else:
        best, best_acc, name = rf, rf_acc, "RandomForest"
        
    print(f"\n[WINNER] {name} with {best_acc*100:.2f}% accuracy")
    print("\nClassification Report:")
    print(classification_report(y_te, best.predict(X_te)))
    
    # ── Save model and labels ──────────────────────────────────────────────────
    joblib.dump(best, MODEL_PATH)
    joblib.dump([str(c) for c in best.classes_], LABELS_PATH)
    print(f"\nSaved model to -> {MODEL_PATH}")
    print(f"Saved labels to -> {LABELS_PATH}")
    print("Model training completed successfully.")

if __name__ == '__main__':
    main()
