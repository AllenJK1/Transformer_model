# Project Understanding: Transformer Model for Financial Time Series

This repository contains code for a Transformer-based neural network model designed for financial time series prediction. The core idea is to use both raw time series data and a set of engineered features as input to the model.

## Key Files and Their Roles:

*   **`trainer.py`**:
    *   **Purpose**: Handles the training of the Transformer model.
    *   **Data Processing**: Loads historical data (e.g., `CADCHFI.csv`, `CADCHFF.csv`), which are already in a 13-column format (OHLC + 9 calculated indicators).
    *   **Windowing**: Extracts 40-period (40-row) windows from the processed data.
    *   **Feature Extraction**: Utilizes `Xfeatures.py` to derive high-level summary features (`class_a` through `class_g`) from each 40-period window.
    *   **Model Architecture**: Defines a Transformer encoder for the main sequence input and parallel dense layers for the side (summary) inputs.
    *   **Training**: Compiles and trains the model using the prepared inputs and labels.
    *   **Output**: Saves the trained model as a `.keras` or `.h5` file.

*   **`inferer.py`**:
    *   **Purpose**: (Original) Intended for making predictions using a pre-trained model.
    *   **Current Limitation**: Assumes the input `live.csv` already contains the 13-column data, which is not realistic for raw live OHLC feeds. This script is being replaced/refactored.

*   **`inference_code.py`**:
    *   **Purpose**: An example/template for loading a saved model and running a prediction with dummy data.

*   **`Xfeatures.py`**:
    *   **Purpose**: Contains functions to extract various statistical and analytical features from a 40-period window of the 13-column data.
    *   **Feature Categories**: Groups features into `class_a` (ADX), `class_b` (Entropy), `class_c` (Gaps/Range), `class_d` (MACD Momentum), `class_e` (Price Action/Trend), `class_f` (Volatility/Noise), and `class_g` (Wick Behavior).
    *   **Output**: Returns seven lists of floats, one for each feature class.

*   **`predictor.py`**:
    *   **Purpose**: A newly created utility to bridge the gap between raw OHLC data and the 13-column format required by the model's training and `Xfeatures.py`.
    *   **Functionality**:
        *   `calculate_indicators(df)`: Takes raw OHLC data and computes the 9 additional technical indicators (Bollinger Bands, ADX, ATR, MACD, candle size, wicks).
        *   `prepare_40_period_data(ohlc_df)`: Takes raw OHLC data, ensures enough historical data (last 75 rows) for indicator calculation, applies `calculate_indicators`, and then extracts the last 40 *fully populated* rows of the 13-column data. This ensures no `NaN` values are passed to the model.

## Data Pipeline for Inference (New Approach):

The intended inference pipeline will be:

1.  **Raw OHLC Data**: Obtain raw OHLC data (e.g., from a live feed or CSV).
2.  **`predictor.py`**: Use `predictor.py`'s `prepare_40_period_data` function to transform the raw OHLC into the 40-row, 13-column format (`main_input` for the model).
3.  **`Xfeatures.py`**: Pass this 40-row, 13-column data to `Xfeatures.py`'s `extract_features` function to get the seven summary feature lists (`side_input_1` to `side_input_7` for the model).
4.  **Model Prediction**: Feed both the 40x13 data and the seven summary feature lists into the loaded Keras model for prediction.

This `GEMINI.md` file summarizes the project's structure and our current task, allowing for quicker context retrieval in future interactions.
