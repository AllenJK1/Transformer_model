# train_model_progressive.py
import os
import re
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Concatenate, Dropout, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention, Add
)

# --- Google Drive mount (safe to run on Colab; no-op elsewhere) ---
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:  # not in Colab
    IN_COLAB = False

# --- Paths ---
BASE_DIR = '/content/drive/MyDrive/USDPKR_training' if IN_COLAB else './USDPKR_training'
SNAP_DIR = os.path.join(BASE_DIR, 'snapshots')      # every 25 epochs
BACKUP_DIR = os.path.join(BASE_DIR, 'backup_state') # per-epoch crash-safe state
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_NAME = 'USDPKRT2'

os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Data prep helpers ---
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

# --- Model ---
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

# --- Callback to save a full model snapshot every N epochs ---
class EveryNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, save_every: int, filepath_template: str, keep_last: int = 10):
        super().__init__()
        self.save_every = int(save_every)
        self.filepath_template = filepath_template
        self.keep_last = int(keep_last)
        self._pattern = re.compile(r"ep(\d+).keras$")

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num % self.save_every == 0:
            path = self.filepath_template.format(epoch=epoch_num)
            self.model.save(path)
            self._prune_old_snapshots()

    def _prune_old_snapshots(self):
        files = [f for f in os.listdir(SNAP_DIR) if f.endswith('.keras')]
        pairs = []
        for f in files:
            m = self._pattern.search(f)
            if m:
                pairs.append((int(m.group(1)), f))
        pairs.sort()
        if len(pairs) > self.keep_last:
            for _, fname in pairs[:-self.keep_last]:
                try:
                    os.remove(os.path.join(SNAP_DIR, fname))
                except FileNotFoundError:
                    pass

# --- Callback to monitor val_accuracy in first 10 epochs ---
class EarlySeedRestart(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, min_change=1e-3):
        super().__init__()
        self.patience = patience
        self.min_change = min_change
        self.val_accs = []
        self.should_restart = False

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return
        if 'val_accuracy' not in logs: return

        self.val_accs.append(logs['val_accuracy'])

        if len(self.val_accs) >= self.patience:
            diffs = [abs(self.val_accs[i+1] - self.val_accs[i]) for i in range(len(self.val_accs)-1)]
            if all(d < self.min_change for d in diffs):
                print("⚠️ Validation accuracy stagnant for first 10 epochs. Triggering restart with new seed.")
                self.should_restart = True
                self.model.stop_training = True

if __name__ == "__main__":
    DATA_PATH = "/content/data/USDPKR_samples.pkl"
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    all_samples = data["samples"]
    y_labels = data["labels"]

    all_samples, y_labels = shuffle(all_samples, y_labels, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(all_samples, y_labels, test_size=0.2, random_state=42)

    train_inputs = prepare_model_inputs(X_train)
    test_inputs = prepare_model_inputs(X_test)

    EPOCHS = 1000
    BATCH_SIZE = 32

    while True:
        model = build_model()
        model.summary()

        backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=BACKUP_DIR)
        snapshot_tmpl = os.path.join(SNAP_DIR, f"{MODEL_NAME}-ep{{epoch:04d}}.keras")
        every25_cb = EveryNEpochs(save_every=25, filepath_template=snapshot_tmpl, keep_last=8)
        csvlog_cb = tf.keras.callbacks.CSVLogger(os.path.join(LOGS_DIR, f"{MODEL_NAME}.csv"), append=True)
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, 'tensorboard'))
        restart_cb = EarlySeedRestart(patience=10, min_change=1e-3)

        history = model.fit(
            train_inputs,
            y_train,
            validation_data=(test_inputs, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=[backup_cb, every25_cb, csvlog_cb, tb_cb, restart_cb],
        )

        if restart_cb.should_restart:
            # Reset seed to new random state
            seed = np.random.randint(0, 1e9)
            tf.keras.utils.set_random_seed(seed)
            print(f"🔄 Restarting training with new seed {seed}")
            continue
        else:
            break

    final_path = os.path.join(BASE_DIR, f"{MODEL_NAME}-final.keras")
    model.save(final_path)
    print(f"✅ Training complete and model saved to: {final_path}")
