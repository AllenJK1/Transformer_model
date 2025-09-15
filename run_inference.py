# run_inference.py
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from predictor import prepare_40_period_data

def get_prediction_for_pair(pair_name: str):
    """
    Orchestrates the complete inference pipeline for a single currency pair.

    This function handles:
    1. Loading the appropriate data file.
    2. Preparing the data using the centralized 'prepare_40_period_data' function.
    3. Loading the corresponding trained model.
    4. Making a prediction.
    5. Returning the result.

    Args:
        pair_name (str): The base name of the currency pair (e.g., "CADCHF").

    Returns:
        float: The raw prediction value (between 0.0 and 1.0), or None if an error occurs.
    """
    print(f"{'='*20} Running Prediction for {pair_name} {'='*20}")

    # --- 1. Define File Paths ---
    model_prefix = f"{pair_name}M"
    csv_filename = f"{pair_name}_otc_sorted.csv"
    model_filename_h5 = f"{model_prefix}.h5"
    model_filename_keras = f"{model_prefix}.keras"

    try:
        # --- 2. Load Raw OHLC Data ---
        if not os.path.exists(csv_filename):
            print(f"Error: Data file not found at '{csv_filename}'")
            return None
        
        ohlc_data = pd.read_csv(csv_filename).tail(75)
        print(f"--- Successfully loaded data from {csv_filename} ---")

        # --- 3. Prepare Data for Model ---
        # This function encapsulates indicator calculation and feature extraction.
        main_input, side_inputs, _ = prepare_40_period_data(ohlc_data)

        # --- 4. Load Trained Model ---
        model = None
        if os.path.exists(model_filename_h5):
            model = tf.keras.models.load_model(model_filename_h5)
            print(f"--- Model Loaded Successfully from {model_filename_h5} ---")
        elif os.path.exists(model_filename_keras):
            model = tf.keras.models.load_model(model_filename_keras)
            print(f"--- Model Loaded Successfully from {model_filename_keras} ---")
        else:
            print(f"Error: Could not find model file '{model_filename_h5}' or '{model_filename_keras}'.")
            return None
        
        # --- 5. Format Inputs for Prediction ---
        # The model expects a batch dimension, so we add one with np.expand_dims.
        main_input_reshaped = np.expand_dims(main_input.to_numpy(), axis=0)
        
        # The side inputs must be in the same order as during training.
        # [class_a, class_f, class_g, class_d, class_e, class_b, class_c]
        side_inputs_ordered = [
            np.expand_dims(side_inputs[0], axis=0), # class_a
            np.expand_dims(side_inputs[5], axis=0), # class_f
            np.expand_dims(side_inputs[6], axis=0), # class_g
            np.expand_dims(side_inputs[3], axis=0), # class_d
            np.expand_dims(side_inputs[4], axis=0), # class_e
            np.expand_dims(side_inputs[1], axis=0), # class_b
            np.expand_dims(side_inputs[2], axis=0)  # class_c
        ]

        # --- 6. Make Prediction ---
        prediction = model.predict([main_input_reshaped] + side_inputs_ordered)
        
        # The model returns a 2D array, e.g., [[0.75]], so we extract the scalar value.
        prediction_value = prediction[0][0]
        return prediction_value

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
    # --- Define the list of pairs you want to get predictions for ---
    pairs_to_predict = ["CADCHF", "USDDZD", "USDPHP", "USDPKR"]
    
    predictions = {}

    for pair in pairs_to_predict:
        prediction_value = get_prediction_for_pair(pair)
        predictions[pair] = prediction_value
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
