import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Concatenate, Dropout, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention, Add
)
from Xfeatures import extract_features

# -------------------------------
# Data Processing
# -------------------------------

def has_asterisks(window):
    return window.astype(str).apply(lambda x: x.str.contains(r"\*{6}")).any().any()

def get_valid_windows(df, window_size=40):
    valid_windows = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        if not has_asterisks(window):
            valid_windows.append(window)
    return valid_windows

def extract_all_structured_samples(file_path, window_size=40):
    df = pd.read_csv(file_path)
    valid_windows = get_valid_windows(df, window_size)
    structured_samples = []

    for window in valid_windows:
        if 'time' in window.columns:
            window = window.drop(columns=['time'])

        class_a, class_b, class_c, class_d, class_e, class_f, class_g = extract_features(window)

        sample = {
            'class_a': np.array(class_a, dtype=np.float32),
            'class_b': np.array(class_b, dtype=np.float32),
            'class_c': np.array(class_c, dtype=np.float32),
            'class_d': np.array(class_d, dtype=np.float32),
            'class_e': np.array(class_e, dtype=np.float32),
            'class_f': np.array(class_f, dtype=np.float32),
            'class_g': np.array(class_g, dtype=np.float32),
            'class_x': window.to_numpy(dtype=np.float32)  # shape: (40, 13)
        }

        structured_samples.append(sample)

    return structured_samples

def prepare_model_inputs(samples):
    return {
        'main_input': np.stack([s['class_x'] for s in samples]),
        'side_input_1': np.stack([s['class_a'] for s in samples]),
        'side_input_2': np.stack([s['class_f'] for s in samples]),
        'side_input_3': np.stack([s['class_g'] for s in samples]),
        'side_input_4': np.stack([s['class_d'] for s in samples]),
        'side_input_5': np.stack([s['class_e'] for s in samples]),
        'side_input_6': np.stack([s['class_b'] for s in samples]),
        'side_input_7': np.stack([s['class_c'] for s in samples]),
    }

# -------------------------------
# Model Definition
# -------------------------------

def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def build_model():
    # Main sequence input
    main_input = Input(shape=(40, 13), name='main_input')
    x_main = transformer_encoder(main_input)
    x_main = GlobalAveragePooling1D()(x_main)
    x_main = Dense(560, activation='relu')(x_main)
    x_main = Dense(280, activation='relu')(x_main)
    x_main = Dense(140, activation='relu')(x_main)

    # Side inputs
    side_inputs = []
    side_processed = []
    side_shapes = [4, 6, 5, 5, 8, 3, 3]

    for i, shape in enumerate(side_shapes):
        input_i = Input(shape=(shape,), name=f'side_input_{i+1}')
        d1 = Dense(shape, activation='relu')(input_i)
        d2 = Dense(max(1, shape // 2), activation='relu')(d1)
        d_combined = Concatenate()([input_i, d2])
        side_inputs.append(input_i)
        side_processed.append(d_combined)

    x_side = Concatenate()(side_processed)
    x_side = Dense(14, activation='relu')(x_side)

    # Merge all
    x = Concatenate()([x_main, x_side])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(4, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[main_input] + side_inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# Training Script
# -------------------------------

def main():
    # Load and label data
    samples_0 = extract_all_structured_samples("CADCHFI.csv")  # label 0
    samples_1 = extract_all_structured_samples("CADCHFF.csv")  # label 1

    all_samples = samples_0 + samples_1
    y_labels = np.array([0] * len(samples_0) + [1] * len(samples_1), dtype=np.float32)

    # Shuffle and split
    all_samples, y_labels = shuffle(all_samples, y_labels, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(all_samples, y_labels, test_size=0.2, random_state=42)

    # Prepare input dictionaries
    train_inputs = prepare_model_inputs(X_train)
    test_inputs = prepare_model_inputs(X_test)

    # Build and train model
    model = build_model()
    model.summary()

    model.fit(
        train_inputs,
        y_train,
        validation_data=(test_inputs, y_test),
        epochs=60,
        batch_size=32,
        verbose=1
    )

    # Optional: Save model
    model.save("binary_classifier_model.h5")
    print("✅ Training complete and model saved.")

if __name__ == "__main__":
    main()
