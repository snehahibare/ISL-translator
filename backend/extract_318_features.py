import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import numpy as np
import mediapipe as mp
import re

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    """
    Returns a flat 258-element vector from MediaPipe Holistic output.
    Layout: Pose (33*4) + Left Hand (21*3) + Right Hand (21*3).
    Zero-padded when a landmark group is not detected.
    """
    pose = (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    lh   = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3))
    rh   = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, lh, rh])


def clean_action_name(folder_name):
    """
    Normalises a raw folder name into a consistent snake_case action label.
    Example: '48. Good Morning' -> 'good_morning'
    """
    name = re.sub(r'^\d+\.\s*', '', folder_name)
    return name.replace(' ', '_').lower()


def process_video(video_path, save_dir):
    """
    Reads a video file, samples exactly 30 frames uniformly, runs
    MediaPipe Holistic on each frame, and saves the resulting 258-d
    keypoint vector as individual .npy files inside save_dir.

    Skips the video silently if the directory is already fully populated.
    """
    if os.path.exists(os.path.join(save_dir, "29.npy")):
        return

    cap    = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return

    # Uniform temporal sampling — keeps sequence length fixed regardless of video fps
    seq_length     = 30
    idx_arr        = np.linspace(0, len(frames) - 1, seq_length).astype(int)
    selected_frames = [frames[i] for i in idx_arr]

    os.makedirs(save_dir, exist_ok=True)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame_num, frame in enumerate(selected_frames):
            try:
                img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results  = holistic.process(img_rgb)
                keypoints = extract_keypoints(results)
                np.save(os.path.join(save_dir, str(frame_num)), keypoints)
            except Exception:
                # If a frame cannot be processed, write a zero vector so the
                # sequence stays complete and downstream code does not break.
                np.save(os.path.join(save_dir, str(frame_num)), np.zeros(258))


def main():
    INCLUDE_DIR   = r"C:\Users\sumeet\Desktop\resume projects\ISL-translator\data\include"
    PROCESSED_DIR = r"C:\Users\sumeet\Desktop\resume projects\ISL-translator\data\processed_318"

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    tasks = []

    print("\n[1/3] Scanning the raw 318-sign dataset...")
    for category in os.listdir(INCLUDE_DIR):
        cat_path = os.path.join(INCLUDE_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        for sign_folder in os.listdir(cat_path):
            sign_path = os.path.join(cat_path, sign_folder)
            if not os.path.isdir(sign_path):
                continue

            action_name = clean_action_name(sign_folder)
            videos = [v for v in os.listdir(sign_path)
                      if v.endswith(('.MOV', '.mp4', '.avi', '.mov', '.MP4'))]

            for seq_num, video_file in enumerate(videos):
                video_path = os.path.join(sign_path, video_file)
                save_dir   = os.path.join(PROCESSED_DIR, action_name, str(seq_num))
                tasks.append((video_path, save_dir))

    print(f"Found {len(tasks)} videos to process.")
    print(f"Output directory: {PROCESSED_DIR}\n")
    print("[2/3] Running MediaPipe frame extraction (this will take a while)...")

    for i, (v_path, s_dir) in enumerate(tasks):
        if i % 10 == 0:
            print(f"  Progress: {i} / {len(tasks)}")
        try:
            process_video(v_path, s_dir)
        except Exception as e:
            print(f"  Error on {v_path}: {e}")

    print("\n[3/3] Extraction complete. Run train_advanced_lstm.py to train the model.")


if __name__ == "__main__":
    main()
