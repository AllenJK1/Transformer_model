# main.py
from classmodule import get_prediction

# List of pairs you want to get predictions for
pairs_to_predict = ["CADCHF", "USDDZD", "USDPHP", "USDPKR"]

predictions = {}

for pair in pairs_to_predict:
    prediction_value = get_prediction(pair)
    predictions[pair] = prediction_value
    
    if prediction_value is not None:
        print(f"--- Prediction Result for {pair} ---")
        print(f"Raw Prediction: {prediction_value}")
        result = 'UP' if prediction_value > 0.5 else 'DOWN'
        print(f"Final result for {pair}: {result} ({1 if result == 'UP' else 0})")
    else:
        print(f"Could not get a prediction for {pair}.")
    print("-" * 30)

# At the end, you can access all predictions like this:
print("\n--- All Predictions ---")
for pair, value in predictions.items():
    if value is not None:
        print(f"{pair}: {value} -> {'UP' if value > 0.5 else 'DOWN'}")
    else:
        print(f"{pair}: Failed to get prediction")
