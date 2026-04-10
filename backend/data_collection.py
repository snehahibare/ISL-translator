import cv2
import numpy as np
import os
import time
import mediapipe as mp

# Fixing Python 3.13 specific import issue for mediapipe
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# Step 1: Folders set karna jaha data save hoga
DATA_PATH = os.path.join(os.getcwd(), 'Custom_Data') 

# Step 2: Konse signs hum record kar rahe hain (Start me 3 testing ke liye)
actions = np.array(['namaste', 'paani', 'help'])

# Har sign (action) ki 30 videos banayenge
no_sequences = 30
# Har video me humara camera 30 frames record karega (1 second approx)
sequence_length = 30

# Numpy Array Directories create karte hain
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Function: Haath ke 21 points ki XYZ values nikal ke flatten karna
def extract_keypoints(results):
    lh = np.zeros(21*3) # Left hand (21 points * XYZ)
    rh = np.zeros(21*3) # Right hand (21 points * XYZ)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[i].classification[0].label
            kps = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if hand_type == "Left":
                lh = kps
            else:
                rh = kps
    # Total 126 coordinate data points per frame!
    return np.concatenate([lh, rh])

cap = cv2.VideoCapture(0)
print("\n[AI] Apna Custom Data Recording start ho raha hai!")

# Setup MediaPipe instance
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    # 3 Layers of Loops: Signs -> Videos -> Frames
    for action in actions:
        for sequence in range(no_sequences):
            # Frame wise iteration
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                # OpenCV feed ko MediaPipe color format me bheja
                imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(imageRGB)                 

                # Draw Map Skeleton
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            frame, hand, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                
                # IMPORTANT LOGIC: Screen pe Text / UI For the User
                if frame_num == 0: 
                    # Jab naya sequence shuru ho, user ko 2 second ka gap do position lene ke liye
                    cv2.putText(frame, 'GET READY...', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting Data for [{action}] - Video {sequence}/30', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('ISL AI - Custom Dataset Builder', frame)
                    cv2.waitKey(2000) # 2 second buffer gap
                else: 
                    # Baaki frames me directly batao recording on hai
                    cv2.putText(frame, f'RECORDING: [{action}] - Video {sequence}/30', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('ISL AI - Custom Dataset Builder', frame)
                
                # MediaPipe Data se Numpy arrays export karo
                keypoints = extract_keypoints(results)
                
                # Numpy file (frame_num.npy) save karo correct folder sequence me 
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == 27: # Press Esc to break
                    break

cap.release()
cv2.destroyAllWindows()
