import pandas as pd
import numpy as np
import tensorflow as tf
from Xfeatures import extract_features

def get_prediction(pair_name: str):
    """
    Main function to get a prediction for a given currency pair.

    Args:
        pair_name (str): The base name of the currency pair (e.g., "CADCHF").

    Returns:
        int: 1 for UP, 0 for DOWN, or None if an error occurs.
    """

    # --- Internal Helper Functions ---

    def _calculate_indicators(df):
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
        df['bbwidth'] = df['bb_upper'] - df['bb_lower']
        # ATR components
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
        df['low_close'] = np.abs(df['low'].shift(1) - df['low'])
        df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        # ADX components
        df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), df['high'] - df['high'].shift(1), 0)
        df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), df['low'].shift(1) - df['low'], 0)
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

    def _prepare_40_period_data(ohlc_df: pd.DataFrame):
        required_ohlc_cols = ['open', 'high', 'low', 'close']
        if not all(col in ohlc_df.columns for col in required_ohlc_cols):
            raise ValueError("Input DataFrame must contain 'open', 'high', 'low', 'close' columns.")
        
        df = ohlc_df[required_ohlc_cols].copy()
        df = df.tail(75).reset_index(drop=True)
        df = _calculate_indicators(df)
        
        feature_cols = [
            'open', 'high', 'low', 'close', 'bbwidth', 'adx',
            'candle_size', 'upper_wick', 'lower_wick', 'atr',
            'macd_line', 'signal_line', 'macd_histogram'
        ]
        
        df_processed = df.dropna()
        if len(df_processed) < 40:
            raise ValueError(f"Not enough rows after indicator calculation. Found {len(df_processed)}, need 40.")
            
        final_df = df_processed.tail(40)
        class_a, class_b, class_c, class_d, class_e, class_f, class_g = extract_features(final_df)
        main_input = final_df[feature_cols]
        side_inputs = [class_a, class_b, class_c, class_d, class_e, class_f, class_g]
        return main_input, side_inputs

    # --- Main Execution Logic ---

    model_prefix = f"{pair_name}M"
    csv_filename = f"{pair_name}_otc_sorted.csv"
    model_filename_h5 = f"{model_prefix}.h5"
    model_filename_keras = f"{model_prefix}.keras"

    print(f"{'='*20} Running Prediction for {pair_name} {'='*20}")

    try:
        # 1. Load Data
        try:
            ohlc_data = pd.read_csv(csv_filename).tail(75)
            print(f"--- Successfully loaded data from {csv_filename} ---")
        except FileNotFoundError:
            print(f"Error: Data file '{csv_filename}' not found.")
            return None

        # 2. Prepare Data
        main_input, side_inputs = _prepare_40_period_data(ohlc_data)

        # 3. Load Model
        model = None
        try:
            model = tf.keras.models.load_model(model_filename_h5)
            print(f"--- Model Loaded Successfully from {model_filename_h5} ---")
        except (OSError, IOError):
            try:
                model = tf.keras.models.load_model(model_filename_keras)
                print(f"--- Model Loaded Successfully from {model_filename_keras} ---")
            except (OSError, IOError):
                print(f"Error: Could not load model '{model_filename_h5}' or '{model_filename_keras}'.")
                return None
        
        # 4. Prepare Inputs for Prediction
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

        # 5. Make Prediction
        predictions = model.predict([main_input_reshaped] + side_inputs_ordered)
        prediction_value = predictions[0][0]

        return prediction_value

    except (ValueError, IndexError) as e:
        print(f"An error occurred during data preparation: {e}")
        return None
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
        return None
