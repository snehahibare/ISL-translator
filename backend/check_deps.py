import sys
try:
    import mediapipe as mp
    print("MediaPipe OK:", mp.__version__)
except Exception as e:
    print("MediaPipe ERROR:", e)

try:
    import google.protobuf
    print("Protobuf OK:", google.protobuf.__version__)
except Exception as e:
    print("Protobuf ERROR:", e)

try:
    import tensorflow as tf
    print("TensorFlow OK:", tf.__version__)
except Exception as e:
    print("TF ERROR:", e)
