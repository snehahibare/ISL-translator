import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.utils import to_categorical
    from keras.optimizers import Adam

DATA_PATH       = r"C:\Users\sumeet\Desktop\resume projects\ISL-translator\data\processed_318"
SEQUENCE_LENGTH = 30

try:
    actions = np.array(sorted([f for f in os.listdir(DATA_PATH)
                                if os.path.isdir(os.path.join(DATA_PATH, f))]))
    print(f"{len(actions)} sign classes detected.")
except FileNotFoundError:
    print("Error: Processed data folder not found. Run extract_318_features.py first.")
    exit()

label_map = {label: num for num, label in enumerate(actions)}

label_map_path = os.path.join(os.path.dirname(__file__), '..', 'label_map.pkl')
with open(label_map_path, 'wb') as f:
    pickle.dump({'label_map': label_map, 'actions': actions}, f)
print("Label map saved.")

# Cache paths — loading 318 signs from disk takes several minutes,
# so we serialise the full numpy arrays on first run.
CACHE_X = os.path.join(os.path.dirname(__file__), '..', 'cache_X_v2.npy')
CACHE_Y = os.path.join(os.path.dirname(__file__), '..', 'cache_y_v2.npy')

if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    print("Found existing cache. Loading...")
    X        = np.load(CACHE_X)
    y_labels = np.load(CACHE_Y)
    print(f"Loaded {X.shape[0]} sequences from cache.")
else:
    print("No cache found. Loading sequences from disk (this may take a few minutes)...")
    sequences, labels = [], []
    skipped = 0

    for i, action in enumerate(actions):
        if i % 30 == 0:
            print(f"  Loading: {i}/{len(actions)} classes...")

        action_path    = os.path.join(DATA_PATH, action)
        available_seqs = sorted(
            [s for s in os.listdir(action_path)
             if os.path.isdir(os.path.join(action_path, s))],
            key=lambda x: int(x) if x.isdigit() else 0
        )

        for seq_folder in available_seqs:
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                try:
                    res = np.load(os.path.join(action_path, seq_folder, f"{frame_num}.npy"))
                    window.append(res)
                except Exception:
                    pass

            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])
            else:
                skipped += 1

    X        = np.array(sequences)
    y_labels = np.array(labels)

    print("Saving cache...")
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y_labels)
    print(f"Done. Loaded {X.shape[0]} sequences (skipped {skipped} incomplete).")


def augment_sequence(seq):
    """
    Generates a set of perturbed copies of a training sequence.

    Strategies applied:
      - Gaussian noise at three magnitudes  (models hand tremor)
      - Uniform scale jitter                (models signer distance)
      - Temporal roll at +/-1 and +/-2     (models signing speed variance)
      - Left/right hand swap               (mirrors the sign horizontally)

    The left-hand keypoints occupy indices [132:195] and right-hand [195:258]
    in the 258-feature vector, matching the layout from extract_keypoints().
    """
    augmented = []

    for sigma in [0.005, 0.01, 0.02]:
        augmented.append(seq + np.random.normal(0, sigma, seq.shape))

    for scale in [0.93, 0.97, 1.03, 1.07]:
        augmented.append(seq * scale)

    for shift in [1, 2, -1, -2]:
        augmented.append(np.roll(seq, shift, axis=0))

    flipped = seq.copy()
    flipped[:, 132:195], flipped[:, 195:258] = seq[:, 195:258].copy(), seq[:, 132:195].copy()
    augmented.append(flipped)

    return augmented


print("Applying data augmentation...")
X_list = [X]
y_list = [y_labels]

for i in range(len(X)):
    for aug in augment_sequence(X[i]):
        X_list.append(aug.reshape(1, *aug.shape))
        y_list.append(np.array([y_labels[i]]))

X_aug = np.concatenate(X_list, axis=0)
y_aug = np.concatenate(y_list, axis=0)
print(f"Dataset expanded: {X.shape[0]} -> {X_aug.shape[0]} sequences "
      f"({X_aug.shape[0] // X.shape[0]}x multiplier).")

# One-hot encode labels for categorical cross-entropy
y_cat = to_categorical(y_aug, num_classes=len(actions)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_cat, test_size=0.1, random_state=42, stratify=y_aug
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Input shape: {X_train.shape[1]} frames x {X_train.shape[2]} keypoints/frame")

# Three stacked LSTM layers with progressively decreasing width
# BatchNorm after each recurrent block stabilises training at high feature counts
model = Sequential([
    LSTM(256, return_sequences=True, activation='tanh',
         input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(512, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(256, return_sequences=False, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)
model.summary()

# Stop training if validation accuracy does not improve for 20 epochs
early_stop = EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Persist only the best checkpoint observed across all epochs
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(os.path.dirname(__file__), '..', 'advanced_action_model.keras'),
    save_best_only=True,
    monitor='val_categorical_accuracy',
    mode='max',
    verbose=1
)

# Halve the learning rate when validation loss plateaus for 8 epochs
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-5,
    verbose=1
)

print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, model_checkpoint, reduce_lr],
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*50}")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}")
print(f"Model saved. Total signs learned: {len(actions)}")
