import pandas as pd
import numpy as np
import tensorflow as tf
from Xfeatures import extract_features

def calculate_indicators(df):
    # Same logic as your class.py
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
    df['bbwidth'] = df['bb_upper'] - df['bb_lower']

    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'].shift(1) - df['low'])
    df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                         df['high'] - df['high'].shift(1), 0)
    df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                         df['low'].shift(1) - df['low'], 0)

    df['TR_smooth'] = df['TR'].ewm(span=14, adjust=False).mean()
    df['+DM_smooth'] = df['+DM'].ewm(span=14, adjust=False).mean()
    df['-DM_smooth'] = df['-DM'].ewm(span=14, adjust=False).mean()

    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    df['DX'] = (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['adx'] = df['DX'].rolling(window=14).mean()

    df['candle_size'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df['close']
    df['lower_wick'] = df['close'] - df['low']

    df['atr'] = df['TR'].ewm(span=14, adjust=False).mean()

    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['signal_line']
    return df

def prepare_live_sample(csv_path):
    # 1. Load the last 75 rows (enough for all indicators to be calculated)
    df = pd.read_csv(csv_path)
    if 'time' in df.columns:
        df = df[['time', 'open', 'high', 'low', 'close']]
    else:
        df.columns = ['open', 'high', 'low', 'close']
    df = df.tail(75).reset_index(drop=True)

    # 2. Calculate all indicators
    df = calculate_indicators(df)

    # 3. Take the last 40 fully populated rows (dropna)
    df_ind = df.drop(columns=['time'], errors='ignore')
    last_40 = df_ind.tail(40).dropna().copy()
    if len(last_40) < 40:
        raise ValueError("Not enough fully populated rows for inference (need 40).")
    # Ensure order of columns matches model's training
    feature_cols = [
        'open', 'high', 'low', 'close', 'bbwidth', 'adx',
        'candle_size', 'upper_wick', 'lower_wick', 'atr',
        'macd_line', 'signal_line', 'macd_histogram'
    ]
    last_40 = last_40[feature_cols]
    return last_40

def main():
    model_path = "as.keras"        # Your model path
    live_csv_path = "live.csv"     # Your live OHLC file (update as needed)

    # Prepare sample for inference
    df_40 = prepare_live_sample(live_csv_path)

    # Use feature extraction (as in training)
    class_a, class_b, class_c, class_d, class_e, class_f, class_g = extract_features(df_40)

    # Prepare input dict (as in trainer.py)
    inputs = {
        'main_input': np.expand_dims(df_40.to_numpy(dtype=np.float32), axis=0),
        'side_input_1': np.expand_dims(np.array(class_a, dtype=np.float32), axis=0),
        'side_input_2': np.expand_dims(np.array(class_f, dtype=np.float32), axis=0),
        'side_input_3': np.expand_dims(np.array(class_g, dtype=np.float32), axis=0),
        'side_input_4': np.expand_dims(np.array(class_d, dtype=np.float32), axis=0),
        'side_input_5': np.expand_dims(np.array(class_e, dtype=np.float32), axis=0),
        'side_input_6': np.expand_dims(np.array(class_b, dtype=np.float32), axis=0),
        'side_input_7': np.expand_dims(np.array(class_c, dtype=np.float32), axis=0),
    }

    # Load Keras model
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(inputs)
    print(f"🔍 Prediction: {prediction[0][0]:.4f}")

if __name__ == "__main__":
    main()
