# Install gdown (if not already installed)
!pip install -q gdown

# Google Drive folder link
folder_url = "https://drive.google.com/drive/folders/1qzHTs66MTAcioaE-RraRCdih4RBwMXd5?usp=drive_link"

# Download entire folder recursively into "data" directory
!gdown --folder "{folder_url}" -O data
import pandas as pd
import numpy as np
from Xfeatures import extract_features

WINDOW_SIZE = 40

def has_asterisks(window):
    return window.astype(str).apply(lambda x: x.str.contains(r"\*{9}")).any().any()

def get_valid_windows(df, window_size=WINDOW_SIZE):
    valid_windows = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        if not has_asterisks(window):
            valid_windows.append(window)
    return valid_windows

def extract_all_structured_samples(file_path, label, window_size=WINDOW_SIZE):
    df = pd.read_csv(file_path)
    valid_windows = get_valid_windows(df, window_size)
    samples = []

    for window in valid_windows:
        if 'time' in window.columns:
            window = window.drop(columns=['time'])

        class_a, class_b, class_c, class_d, class_e, class_f, class_g = extract_features(window)

        samples.append({
            'class_a': np.array(class_a, dtype=np.float32),
            'class_b': np.array(class_b, dtype=np.float32),
            'class_c': np.array(class_c, dtype=np.float32),
            'class_d': np.array(class_d, dtype=np.float32),
            'class_e': np.array(class_e, dtype=np.float32),
            'class_f': np.array(class_f, dtype=np.float32),
            'class_g': np.array(class_g, dtype=np.float32),
            'class_x': window.to_numpy(dtype=np.float32),
            'label': label
        })

    return samples

if __name__ == "__main__":
    # Prepare both losing and winning samples
    samples_0 = extract_all_structured_samples("/content/data/CADCHF_losss_indicators.csv", label=0)
    samples_1 = extract_all_structured_samples("/content/data/CADCHF_winn_indicators.csv", label=1)

    all_samples = samples_0 + samples_1

    # Convert dict-of-arrays into arrays for np.savez
    np.savez_compressed(
        "CADCHF_samples.npz",
        class_a=np.stack([s['class_a'] for s in all_samples]),
        class_b=np.stack([s['class_b'] for s in all_samples]),
        class_c=np.stack([s['class_c'] for s in all_samples]),
        class_d=np.stack([s['class_d'] for s in all_samples]),
        class_e=np.stack([s['class_e'] for s in all_samples]),
        class_f=np.stack([s['class_f'] for s in all_samples]),
        class_g=np.stack([s['class_g'] for s in all_samples]),
        class_x=np.stack([s['class_x'] for s in all_samples]),
        labels=np.array([s['label'] for s in all_samples], dtype=np.float32)
    )

    print("✅ Samples saved to CADCHF_samples.npz")
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Concatenate, Dropout, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention, Add
)

def prepare_model_inputs(data):
    return {
        'main_input': data['class_x'],
        'side_input_1': data['class_a'],
        'side_input_2': data['class_f'],
        'side_input_3': data['class_g'],
        'side_input_4': data['class_d'],
        'side_input_5': data['class_e'],
        'side_input_6': data['class_b'],
        'side_input_7': data['class_c'],
    }

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
    main_input = Input(shape=(40, 13), name='main_input')
    x_main = transformer_encoder(main_input)
    x_main = GlobalAveragePooling1D()(x_main)
    x_main = Dense(560, activation='relu')(x_main)
    x_main = Dense(280, activation='relu')(x_main)
    x_main = Dense(140, activation='relu')(x_main)

    side_inputs = []
    side_processed = []
    side_shapes = [4, 6, 5, 5, 8, 3, 3]
    for i, shape in enumerate(side_shapes):
        input_i = Input(shape=(shape,), name=f'side_input_{i+1}')
        d1 = Dense(shape * 4, activation='relu')(input_i)
        d2 = Dense(shape * 2, activation='relu')(d1)
        d_combined = Concatenate()([input_i, d2])
        side_inputs.append(input_i)
        side_processed.append(d_combined)

    x_side = Concatenate()(side_processed)
    x_side = Dense(14, activation='relu')(x_side)

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

if __name__ == "__main__":
    # Load preprocessed data
    data = np.load("CADCHF_samples.npz")

    labels = data['labels']
    all_data = {
        'class_a': data['class_a'],
        'class_b': data['class_b'],
        'class_c': data['class_c'],
        'class_d': data['class_d'],
        'class_e': data['class_e'],
        'class_f': data['class_f'],
        'class_g': data['class_g'],
        'class_x': data['class_x']
    }

    # Safe shuffle (keep all arrays & labels aligned)
    sample_count = len(labels)
    indices = np.arange(sample_count)
    np.random.shuffle(indices)
    for k in all_data:
        all_data[k] = all_data[k][indices]
    labels = labels[indices]

    # Train/test split
    train_size = int(0.8 * sample_count)
    train_data = {k: v[:train_size] for k, v in all_data.items()}
    test_data = {k: v[train_size:] for k, v in all_data.items()}
    y_train, y_test = labels[:train_size], labels[train_size:]

    # Prepare inputs
    train_inputs = prepare_model_inputs(train_data)
    test_inputs = prepare_model_inputs(test_data)

    # Build & train
    model = build_model()
    model.summary()
    model.fit(train_inputs, y_train,
              validation_data=(test_inputs, y_test),
              epochs=250, batch_size=32)
    model.save("CADCHFT.h5")
    print("✅ Model trained and saved.")
