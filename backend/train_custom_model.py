import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = r"C:\Users\sumeet\Desktop\resume projects\ISL-translator\data\processed"

try:
    actions = np.array(sorted([f for f in os.listdir(DATA_PATH)
                                if os.path.isdir(os.path.join(DATA_PATH, f))]))
    print(f"{len(actions)} sign classes found.")
except FileNotFoundError:
    print("Error: Data folder not found. Run data_collection.py first.")
    exit()

if len(actions) == 0:
    print("Error: No sign folders found in the data directory.")
    exit()

no_sequences    = 30
sequence_length = 30
label_map       = {label: num for num, label in enumerate(actions)}

# Persist the label map so inference scripts use the same class ordering
label_map_path = os.path.join(os.path.dirname(__file__), '..', 'label_map.pkl')
with open(label_map_path, 'wb') as f:
    pickle.dump({'label_map': label_map, 'actions': actions}, f)
print("Label map saved.")

# Load the .npy keypoint sequences written by data_collection.py
sequences, labels = [], []
skipped = 0
print("Loading training data...")

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            except Exception:
                pass
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            skipped += 1

if len(sequences) == 0:
    print("Error: No valid sequences found. Check your data directory.")
    exit()

X = np.array(sequences)
y = np.array(labels)
print(f"Loaded {X.shape[0]} sequences with {X.shape[2]} features per frame.")
if skipped > 0:
    print(f"Skipped {skipped} incomplete sequences.")


def augment_sequence(seq):
    """
    Returns several augmented copies of a single training sequence.
    Augmentations (noise, scaling, time-shift) help the model generalise
    to real-world variation in signing speed and distance from camera.
    """
    augmented = []

    # Gaussian noise — simulates natural hand tremor
    for _ in range(2):
        noise = np.random.normal(0, 0.01, seq.shape)
        augmented.append(seq + noise)

    # Scale jitter — simulates the signer being closer or further from the camera
    for scale in [0.95, 1.05]:
        augmented.append(seq * scale)

    # Temporal shift — simulates slightly different signing speed
    for shift in [1, 2]:
        augmented.append(np.roll(seq, shift, axis=0))

    return augmented


print("Applying data augmentation...")
X_aug_list = [X]
y_aug_list = [y]

for i in range(len(X)):
    for aug in augment_sequence(X[i]):
        X_aug_list.append(aug.reshape(1, *aug.shape))
        y_aug_list.append(np.array([y[i]]))

X_aug = np.concatenate(X_aug_list, axis=0)
y_aug = np.concatenate(y_aug_list, axis=0)
print(f"Augmented: {X.shape[0]} -> {X_aug.shape[0]} sequences ({X_aug.shape[0] // X.shape[0]}x).")

# Flatten the 3-D sequence array before feeding it to scikit-learn
# Shape: (samples, 30, 258) -> (samples, 7740)
X_flat = X_aug.reshape(X_aug.shape[0], -1)

# Standardise features so gradient-based optimisers converge properly
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_aug, test_size=0.1, random_state=42, stratify=y_aug
)
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples.")

# MLP — captures complex, non-linear feature interactions
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),
    max_iter=800,
    activation='relu',
    solver='adam',
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    batch_size=32,
    verbose=False
)

# Random Forest — complementary ensemble member; robust to noise
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

# Soft-voting ensemble combines both models' class probabilities
model = VotingClassifier(
    estimators=[('mlp', mlp), ('rf', rf)],
    voting='soft',
    n_jobs=-1
)

print("Training ensemble model (MLP + Random Forest)...")
model.fit(X_train, y_train)

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}")
print("\nPer-class accuracy report:")
print(classification_report(y_test, y_pred,
                             target_names=[str(a) for a in actions],
                             zero_division=0))

model_path  = os.path.join(os.path.dirname(__file__), '..', 'action_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Model saved  -> {model_path}")
print(f"Scaler saved -> {scaler_path}")
print(f"Signs learned: {list(actions)}")
