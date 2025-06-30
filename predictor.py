import pandas as pd
import numpy as np
import sys
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

if __name__ == '__main__':
    try:
        # Load the last 75 rows from the specified CSV file
        try:
            ohlc_data = pd.read_csv('CADCHF_otc_sorted.csv').tail(75)
        except FileNotFoundError:
            print("Error: 'CADCHF_otc_sorted.csv' not found. Please ensure the file is in the correct directory.")
            sys.exit(1)
        
        # Set pandas to display all rows
        pd.set_option('display.max_rows', None)

        # The function can raise a ValueError, which is caught below
        result = prepare_40_period_data(ohlc_data)
        
        # The result is a tuple, so it will always be "truthy" if no error was raised
        main_input, side_inputs, time_data = result

        print("--- Verification ---")
        print(f"Main Input Shape: {main_input.shape}")
        if time_data is not None:
            print(f"Time Data Shape: {time_data.shape}")
        else:
            print("Time Data: Not available")
        print(f"Number of Side Input Feature-Sets: {len(side_inputs)}")
        print(f"Lengths of each Feature-Set: {[len(f) for f in side_inputs]}")
        print("-" * 20)

        print("\n--- Main Input (Full 40 Rows) ---")
        print(main_input)
        print("-" * 20)

        print("\n--- Extracted Features (Classes A-G) ---")
        for i, features in enumerate(side_inputs):
            class_name = f"Class {chr(ord('A') + i)}"
            print(f"\n{class_name}:")
            feature_dict = {f"{j+1}": val for j, val in enumerate(features)}
            print(feature_dict)
        print("-" * 20)

    except (ValueError, IndexError) as e:
        print(f"An error occurred: {e}")
        print("This might happen if the dummy data doesn't produce enough valid rows (40) after indicator calculation.")
