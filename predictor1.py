import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from Xfeatures import extract_features

def calculate_indicators(df):
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
    df['bbwidth'] = df['bb_upper'] - df['bb_lower']

    # Average True Range (ATR) components
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'].shift(1) - df['low'])
    df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # ADX components
    df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                         df['high'] - df['high'].shift(1), 0)
    df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                         df['low'].shift(1) - df['low'], 0)

    df['TR_smooth'] = df['TR'].ewm(span=14, adjust=False).mean()
    df['+DM_smooth'] = df['+DM'].ewm(span=14, adjust=False).mean()
    df['-DM_smooth'] = df['-DM'].ewm(span=14, adjust=False).mean()

    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    df['DX'] = (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 98
    df['adx'] = df['DX'].rolling(window=14).mean()

    # Candle features
    df['candle_size'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df['close']
    df['lower_wick'] = df['close'] - df['low']

    # ATR
    df['atr'] = df['TR'].ewm(span=14, adjust=False).mean()

    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['signal_line']

    return df

def prepare_40_period_data(ohlc_df: pd.DataFrame):
    # Ensure OHLC columns are present and in correct order
    required_ohlc_cols = ['open', 'high', 'low', 'close']
    if not all(col in ohlc_df.columns for col in required_ohlc_cols):
        raise ValueError("Input DataFrame must contain 'open', 'high', 'low', 'close' columns.")

    # Keep 'time' column if it exists
    if 'time' in ohlc_df.columns:
        df = ohlc_df[['time'] + required_ohlc_cols].copy()
    else:
        df = ohlc_df[required_ohlc_cols].copy()

    # Take the last 75 rows to ensure enough data for indicator calculation
    df = df.tail(75).reset_index(drop=True)

    # Calculate all indicators
    df = calculate_indicators(df)

    # Define the 13 feature columns (including OHLC)
    feature_cols = [
        'open', 'high', 'low', 'close', 'bbwidth', 'adx',
        'candle_size', 'upper_wick', 'lower_wick', 'atr',
        'macd_line', 'signal_line', 'macd_histogram'
    ]

    # Drop rows with NaN values (due to indicator calculation) and take the last 40
    # This ensures we have 40 fully populated rows
    df_processed = df.dropna()

    if len(df_processed) < 40:
        raise ValueError(f"Not enough fully populated rows after indicator calculation. Found {len(df_processed)}, need 40.")

    # Select the last 40 rows and the required feature columns
    final_df = df_processed.tail(40)

    # Extract summary features using Xfeatures
    class_a, class_b, class_c, class_d, class_e, class_f, class_g = extract_features(final_df)

    # The main input for the model is the 40x13 dataframe
    main_input = final_df[feature_cols]

    # The side inputs are the extracted feature lists
    side_inputs = [class_a, class_b, class_c, class_d, class_e, class_f, class_g]

    # If 'time' column exists, return it alongside the other data
    time_output = final_df['time'] if 'time' in final_df.columns else None
    return main_input, side_inputs, time_output

def run_prediction(model_prefix):
    """
    Runs the prediction pipeline for a given model prefix.
    """
    print(f"{'='*20} Running Prediction for {model_prefix} {'='*20}")
    model_filename_keras = f"{model_prefix}.keras"
    model_filename_h5 = f"{model_prefix}.h5"
    # Derive CSV name by removing the last character (e.g., 'M') from the prefix
    csv_filename = f"{model_prefix[:-1]}_otc_sorted.csv"

    try:
        # Load the last 75 rows from the specified CSV file
        try:
            ohlc_data = pd.read_csv(csv_filename).tail(75)
            print(f"--- Successfully loaded data from {csv_filename} ---")
        except FileNotFoundError:
            print(f"Error: '{csv_filename}' not found. Skipping this model.")
            return  # Exit this function and move to the next model

        # Set pandas to display all rows
        pd.set_option('display.max_rows', None)

        # The function can raise a ValueError, which is caught below
        main_input, side_inputs, time_data = prepare_40_period_data(ohlc_data)

        # Load the trained model
        model = None
        try:
            model = tf.keras.models.load_model(model_filename_h5)
            print(f"--- Model Loaded Successfully from {model_filename_h5} ---")
        except (OSError, IOError):
            try:
                model = tf.keras.models.load_model(model_filename_keras)
                print(f"--- Model Loaded Successfully from {model_filename_keras} ---")
            except (OSError, IOError) as e:
                print(f"Error loading model for {model_prefix}: {e}")
                print(f"Looked for {model_filename_h5} and {model_filename_keras}. Skipping this model.")
                return

        # Prepare inputs for the model
        main_input_reshaped = np.expand_dims(main_input.to_numpy(), axis=0)
        side_inputs_ordered = [
            np.expand_dims(side_inputs[0], axis=0), # class_a
            np.expand_dims(side_inputs[5], axis=0), # class_f
            np.expand_dims(side_inputs[6], axis=0), # class_g
            np.expand_dims(side_inputs[3], axis=0), # class_d
            np.expand_dims(side_inputs[4], axis=0), # class_e
            np.expand_dims(side_inputs[1], axis=0), # class_b
            np.expand_dims(side_inputs[2], axis=0)  # class_c
        ]

        # Make prediction
        predictions = model.predict([main_input_reshaped] + side_inputs_ordered)
        print(f"--- Prediction Result for {model_prefix} ---")
        print(f"Raw Prediction: {predictions[0][0]}")
        if predictions[0][0] > 0.5:
            print("Prediction: UP (1)")
        else:
            print("Prediction: DOWN (0)")

    except (ValueError, IndexError) as e:
        print(f"An error occurred during processing for {model_prefix}: {e}")
        print("This might happen if the data doesn't produce enough valid rows (40) after indicator calculation.")
    except Exception as e:
        print(f"An unexpected error occurred for {model_prefix}: {e}")


if __name__ == '__main__':
    # List of model prefixes to iterate through
    model_prefixes = ['CADCHFM', 'USDDZDM', 'USDPHPM', 'USDPKRM']

    for prefix in model_prefixes:
        run_prediction(prefix)
