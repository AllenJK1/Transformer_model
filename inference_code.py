def main():
    # Example input (replace with your actual data)
    class_x = np.zeros((40, 13), dtype=np.float32)
    class_a = np.zeros((4,), dtype=np.float32)
    class_b = np.zeros((3,), dtype=np.float32)
    class_c = np.zeros((3,), dtype=np.float32)
    class_d = np.zeros((5,), dtype=np.float32)
    class_e = np.zeros((8,), dtype=np.float32)
    class_f = np.zeros((6,), dtype=np.float32)
    class_g = np.zeros((5,), dtype=np.float32)

    # Format as batch of 1
    inputs = {
        'main_input': np.expand_dims(class_x, axis=0),
        'side_input_1': np.expand_dims(class_a, axis=0),
        'side_input_2': np.expand_dims(class_f, axis=0),
        'side_input_3': np.expand_dims(class_g, axis=0),
        'side_input_4': np.expand_dims(class_d, axis=0),
        'side_input_5': np.expand_dims(class_e, axis=0),
        'side_input_6': np.expand_dims(class_b, axis=0),
        'side_input_7': np.expand_dims(class_c, axis=0),
    }

    # Load model
    model = tf.keras.models.load_model("binary_classifier_model.h5")

    # Run prediction
    prediction = model.predict(inputs)
    print(f"🔍 Prediction: {prediction[0][0]:.4f}")
