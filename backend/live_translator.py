import cv2
import numpy as np
import mediapipe as mp
import pickle
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'action_model.pkl')
LABEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'label_map.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')

print("Loading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded.")
except FileNotFoundError:
    print("ERROR: 'action_model.pkl' not found. Run train_custom_model.py first.")
    exit()

# Load the label map saved during training to ensure consistent label ordering
try:
    with open(LABEL_PATH, 'rb') as f:
        label_data = pickle.load(f)
        actions = label_data['actions']
    print(f"{len(actions)} signs loaded.")
except FileNotFoundError:
    print("Warning: label_map.pkl not found, falling back to directory scan.")
    DATA_PATH = r"C:\Users\sumeet\Desktop\resume projects\ISL-translator\data\processed"
    try:
        actions = np.array(sorted([f for f in os.listdir(DATA_PATH)
                                   if os.path.isdir(os.path.join(DATA_PATH, f))]))
    except Exception:
        print("ERROR: Could not determine sign labels.")
        exit()

# Load the feature scaler (may not exist for older models)
scaler = None
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Warning: No scaler found. Predictions may be less accurate.")

# Rolling buffers
sequence    = []   # Holds the last 30 frames of keypoints
sentence    = []   # Accumulates detected signs for display
predictions = []   # Tracks recent prediction indices for a stability check


def extract_keypoints(results):
    """
    Returns a flat 258-dimensional vector from MediaPipe Holistic results.
    Layout: Pose (33*4=132) + Left Hand (21*3=63) + Right Hand (21*3=63)
    This must match the format used during data collection.
    """
    pose = (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    lh   = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3))
    rh   = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, lh, rh])


cap = cv2.VideoCapture(0)
print("\nLive ISL translation started. Press ESC to quit.\n")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # Draw hand skeleton overlays
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0),   thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0),   thickness=2, circle_radius=2)
        )

        # Build the rolling window of keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence[:] = sequence[-30:]

        # Run prediction once we have a full 30-frame window
        if len(sequence) == 30:
            seq_flat = np.array(sequence).reshape(1, -1)
            if scaler is not None:
                seq_flat = scaler.transform(seq_flat)

            try:
                probs          = model.predict_proba(seq_flat)[0]
                prediction_idx = np.argmax(probs)
                confidence     = probs[prediction_idx]
                predictions.append(prediction_idx)

                # Accept a sign only when the last 10 predictions consistently agree
                if (len(predictions) >= 10
                        and np.unique(predictions[-10:])[0] == prediction_idx
                        and confidence > 0.6):
                    current_action = actions[prediction_idx]
                    if not sentence or current_action != sentence[-1]:
                        sentence.append(current_action)

                # Keep the displayed sentence to a reasonable length
                if len(sentence) > 7:
                    sentence = sentence[-7:]

                cv2.putText(frame, f"Confidence: {confidence:.0%}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            except Exception:
                cv2.putText(frame, "Prediction error", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Top bar — shows accumulated sign words
        cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)
        display_text = ' '.join(sentence).upper() if sentence else "SHOW A SIGN..."
        cv2.putText(frame, f"ISL: {display_text}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('ISL Translator — Live', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")
