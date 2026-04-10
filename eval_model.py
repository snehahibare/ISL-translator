import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pickle

try:
    from keras.models import load_model
    from keras.utils import to_categorical
except Exception:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

BASE = os.path.dirname(__file__)

model   = load_model(os.path.join(BASE, 'advanced_action_model.keras'))
label_f = os.path.join(BASE, 'label_map.pkl')
cache_x = os.path.join(BASE, 'cache_X_v2.npy')
cache_y = os.path.join(BASE, 'cache_y_v2.npy')

with open(label_f, 'rb') as f:
    data = pickle.load(f)
actions = data['actions']

X        = np.load(cache_x)
y_labels = np.load(cache_y)
y_cat    = to_categorical(y_labels, num_classes=len(actions))

_, X_test, _, y_test = train_test_split(
    X, y_cat, test_size=0.1, random_state=42, stratify=y_labels
)

print(f"Total signs  : {len(actions)}")
print(f"Test samples : {len(X_test)}")

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc * 100:.2f}%")
print(f"Test loss    : {loss:.4f}")
