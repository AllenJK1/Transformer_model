import os
import asyncio
import configparser
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Calculate indicators
def calculate_indicators(df):
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

# Load settings
def load_settings(path):
    config = configparser.ConfigParser()
    config.read(path)
    assets = [a.strip() for a in config['ASSETS']['asset_list'].split(',') if a.strip()]
    invert = set(p.strip() for p in config['INVERT']['pairs'].split(',') if p.strip())
    follow = set(p.strip() for p in config['FOLLOW']['pairsF'].split(',') if p.strip())
    return config, assets, invert, follow

# Save settings only if changed
def save_settings(path, config, invert, follow):
    config['INVERT']['pairs'] = ','.join(sorted(invert))
    config['FOLLOW']['pairsF'] = ','.join(sorted(follow))
    with open(path, 'w') as configfile:
        config.write(configfile)

# Process a single asset
async def process_asset(asset, model, scaler, invert_list, follow_list):
    try:
        filename = f"{asset.replace('_otc', '')}.csv"
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False

        df = pd.read_csv(filename, header=None, names=['time', 'open', 'high', 'low', 'close'])
        df = calculate_indicators(df).dropna()
        df = df.tail(40)

        feature_cols = ['open', 'high', 'low', 'close', 'bbwidth', 'adx',
                        'candle_size', 'upper_wick', 'lower_wick', 'atr',
                        'macd_line', 'signal_line', 'macd_histogram']
        
        if len(df) < 21:
            print(f"Not enough data for {asset}")
            return False

        sample = df[feature_cols].tail(21).values

        # Scale sample independently
        sample_scaled = scaler.fit_transform(sample)
        sample_scaled = np.expand_dims(sample_scaled, axis=0)  # (1, 21, 13)

        pred = model.predict(sample_scaled, verbose=0)[0][0]

        changed = False

        if pred >= 0.7:
            if asset not in follow_list:
                follow_list.add(asset)
                changed = True
            if asset in invert_list:
                invert_list.discard(asset)
                changed = True
        elif pred <= 0.3:
            if asset not in invert_list:
                invert_list.add(asset)
                changed = True
            if asset in follow_list:
                follow_list.discard(asset)
                changed = True
        else:
            if asset in invert_list or asset in follow_list:
                invert_list.discard(asset)
                follow_list.discard(asset)
                changed = True

        print(f"{asset} → Prediction: {pred:.3f}")

        return changed

    except Exception as e:
        print(f"Error processing {asset}: {e}")
        return False

# Main loop with asyncio
async def main_loop():
    settings_path = 'settingz.ini'
    config, assets, invert_list, follow_list = load_settings(settings_path)

    models = {}
    scalers = {}

    for asset in assets:
        try:
            model_path = f"{asset}C.keras"
            models[asset] = load_model(model_path)
            scalers[asset] = StandardScaler()
            print(f"Loaded model for {asset}")
        except Exception as e:
            print(f"Failed to load model for {asset}: {e}")

    while True:
        changed_any = False
        tasks = []

        # Create async tasks for asset processing
        for asset in assets:
            model = models.get(asset)
            scaler = scalers.get(asset)
            if model and scaler:
                tasks.append(process_asset(asset, model, scaler, invert_list, follow_list))

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # If any result changed, update settings
        if any(results):
            save_settings(settings_path, config, invert_list, follow_list)
            print("Updated settingz.ini.")

        print("Waiting 40 seconds...")
        await asyncio.sleep(40)

# Run async main loop
if __name__ == '__main__':
    asyncio.run(main_loop())
