import tensorflow as tf
from tensorflow import keras
import h5py

model_path = "CADCHFM.keras"

def check_model_version(model_path):
    try:
        # Attempt to read HDF5 file if it's saved in HDF5 format
        with h5py.File(model_path, 'r') as f:
            if 'training_config' in f.attrs:
                print("TensorFlow/Keras version info (HDF5):")
                for key in f.attrs:
                    print(f"{key}: {f.attrs[key]}")
            else:
                print("No training config found in HDF5 attributes.")
    except Exception as e:
        print(f"Not HDF5 format or failed to read with h5py: {e}")

    # Try loading the model using Keras to see if version info appears
    try:
        model = keras.models.load_model(model_path)
        print("\nModel loaded successfully using TensorFlow version:", tf.__version__)
    except Exception as e:
        print("Could not load model:", e)

if __name__ == "__main__":
    check_model_version(model_path)
