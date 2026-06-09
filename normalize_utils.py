import numpy as np

def normalize_landmarks(landmarks, aspect_ratio=1.3333):
    """
    Normalizes a flat list of 63 coordinates (21 landmarks x,y,z).
    - Adjusts x coordinates based on the frame's aspect ratio to ensure aspect-ratio independence.
    - Zeros out the depth (z) coordinates to eliminate high sensor noise in depth estimations.
    - Subtracts the wrist coordinate (landmark 0) from all landmarks (translation invariance).
    - Scales all coordinates by the maximum Euclidean distance from the wrist (scale invariance).
    - Returns a normalized numpy array of shape (1, 63).
    """
    # Convert input to numpy array and reshape to (21, 3)
    landmarks_arr = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    
    # 1. Correct aspect ratio (stretch x coordinate to square coordinate grid)
    # x is column 0
    landmarks_arr[:, 0] = landmarks_arr[:, 0] * aspect_ratio
    
    # 2. Zero out z (depth) to eliminate noise in depth estimation from 2D webcams
    # z is column 2
    landmarks_arr[:, 2] = 0.0
    
    # Landmark 0 is the wrist
    wrist = landmarks_arr[0]
    
    # Subtract wrist position (translation invariance)
    temp_landmarks = landmarks_arr - wrist
    
    # Calculate Euclidean distance from wrist for each landmark
    distances = np.linalg.norm(temp_landmarks, axis=1)
    
    # Find the maximum distance
    max_distance = np.max(distances)
    
    # Scale by maximum distance (scale invariance)
    if max_distance > 0:
        temp_landmarks = temp_landmarks / max_distance
        
    # Flatten back to a flat array and reshape to (1, 63) for model prediction
    return temp_landmarks.flatten().reshape(1, 63)

if __name__ == "__main__":
    # Quick self-test with dummy landmarks (all zeros except wrist)
    dummy_input = [0.0] * 63
    dummy_input[3] = 1.0  # landmark 1 x-coord
    dummy_input[4] = 2.0  # landmark 1 y-coord
    result = normalize_landmarks(dummy_input)
    print("Normalizer Self-Test shape:", result.shape)
    print("Normalizer Self-Test sample values (landmark 1):", result[0, 3:6])
