# main.py
from classmodule import get_prediction

# List of pairs you want to get predictions for
pairs_to_predict = ["CADCHF", "USDDZD", "USDPHP", "USDPKR"]

for pair in pairs_to_predict:
    result = get_prediction(pair)
    if result is not None:
        print(f"Final result for {pair}: {'UP' if result == 1 else 'DOWN'}")
    else:
        print(f"Could not get a prediction for {pair}.")
    print("-" * 30)
