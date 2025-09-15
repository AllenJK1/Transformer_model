# inference.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from predictor import prepare_40_period_data

def load_model_for_pair(pair_name: str):
    """
    Loads the trained Keras model for a specific currency pair.
    It checks for both .h5 and .keras file extensions.

    Args:
        pair_name (str): The base name of the currency pair (e.g., "CADCHF").

    Returns:
        tf.keras.Model: The loaded model, or None if no model file is found.
    """
    model_prefix = f"{pair_name}M"
    model_filename_h5 = f"{model_prefix}.h5"
    model_filename_keras = f"{model_prefix}.keras"

    if os.path.exists(model_filename_h5):
        print(f"--- Loading model from {model_filename_h5} ---")
        return tf.keras.models.load_model(model_filename_h5)
    elif os.path.exists(model_filename_keras):
        print(f"--- Loading model from {model_filename_keras} ---")
        return tf.keras.models.load_model(model_filename_keras)
    else:
        print(f"Error: Could not find model file '{model_filename_h5}' or '{model_filename_keras}'.")
        return None

def get_prediction(pair_name: str, model: tf.keras.Model):
    """
    Generates a prediction for a given currency pair using a pre-loaded model.

    Args:
        pair_name (str): The base name of the currency pair (e.g., "CADCHF").
        model (tf.keras.Model): The trained model to use for prediction.

    Returns:
        float: The raw prediction value (between 0.0 and 1.0), or None on error.
    """
    csv_filename = f"{pair_name}_otc_sorted.csv"
    
    try:
        # 1. Load Raw OHLC Data
        if not os.path.exists(csv_filename):
            print(f"Error: Data file not found at '{csv_filename}'")
            return None
        
        ohlc_data = pd.read_csv(csv_filename).tail(75)
        print(f"--- Successfully loaded data from {csv_filename} ---")

        # 2. Prepare Data for Model using the function from predictor.py
        main_input, side_inputs, _ = prepare_40_period_data(ohlc_data)

        # 3. Format Inputs for Prediction
        main_input_reshaped = np.expand_dims(main_input.to_numpy(), axis=0)
        
        # The side inputs must be in the same order as during training.
        side_inputs_ordered = [
            np.expand_dims(side_inputs[0], axis=0), # class_a
            np.expand_dims(side_inputs[5], axis=0), # class_f
            np.expand_dims(side_inputs[6], axis=0), # class_g
            np.expand_dims(side_inputs[3], axis=0), # class_d
            np.expand_dims(side_inputs[4], axis=0), # class_e
            np.expand_dims(side_inputs[1], axis=0), # class_b
            np.expand_dims(side_inputs[2], axis=0)  # class_c
        ]

        # 4. Make Prediction
        prediction = model.predict([main_input_reshaped] + side_inputs_ordered)
        return prediction[0][0]

    except (ValueError, IndexError) as e:
        print(f"An error occurred during data preparation for {pair_name}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected critical error occurred for {pair_name}: {e}")
        return None

def main():
    """
    Main function to run inference for a predefined list of currency pairs.
    """
    pairs_to_predict = ["CADCHF", "USDDZD", "USDPHP", "USDPKR"]
    predictions = {}

    for pair in pairs_to_predict:
        print(f"{ '='*20} Processing {pair} {'='*20}")
        
        model = load_model_for_pair(pair)
        if model:
            prediction_value = get_prediction(pair, model)
            predictions[pair] = prediction_value
        else:
            predictions[pair] = None # Mark as failed
        
        print("-" * 50)

    # --- Display Final Summary ---
    print("\n--- Final Prediction Summary ---")
    for pair, value in predictions.items():
        if value is not None:
            result = 'UP' if value > 0.5 else 'DOWN'
            print(f"{pair}: {value:.4f} -> {result}")
        else:
            print(f"{pair}: Failed to get prediction.")

if __name__ == "__main__":
    main()
