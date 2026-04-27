import polars as pl
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import sys

# Ensure src module can be imported from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocess import load_and_clean_data, add_rolling_features
from src.models.train import asymmetric_loss

def evaluate_model():
    print("Loading test data and ground truth RUL...")
    test_df = load_and_clean_data('datasets/RUL/test_FD001.txt')
    
    rul_truth = pl.read_csv('datasets/RUL/RUL_FD001.txt', has_header=False, new_columns=['true_RUL'])
    rul_truth = rul_truth.with_columns(pl.Series("engine_id", range(1, rul_truth.height + 1)))

    print("Computing rolling features on test data...")
    test_df = add_rolling_features(test_df, window_size=15)
    
    # We only want to predict and evaluate the LAST cycle for each engine
    last_cycle_df = (
        test_df
        .group_by("engine_id")
        .tail(1) # Gets the last row for each engine
    )
    
    # Join with ground truth
    eval_df = last_cycle_df.join(rul_truth, on="engine_id")
    
    # Define features
    features = [
        'cycle', 
        'sensor_4', 'sensor_11', 
        'sensor_4_rolling_mean', 'sensor_11_rolling_mean', 
        'sensor_4_rolling_std', 'sensor_11_rolling_std'
    ]
    
    X_test = eval_df.select(features).to_pandas()
    y_test = eval_df.select('true_RUL').to_pandas().values.ravel().astype(float)
    
    # Clip ground truth targets for a fair evaluation (Piecewise Linear RUL)
    MAX_RUL = 130
    y_test_clipped = np.clip(y_test, a_min=None, a_max=MAX_RUL)

    print("Loading trained LightGBM model...")
    model_path = 'models/lgb_model.txt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train it first.")
        return
        
    booster = lgb.Booster(model_file=model_path)
    
    print("Running inference...")
    preds = booster.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_clipped, preds))
    score = asymmetric_loss(y_test_clipped, preds)
    
    print("-" * 30)
    print("Test Set Evaluation Metrics (Clipped to 130)")
    print("-" * 30)
    print(f"Test RMSE: {rmse:.2f} cycles")
    print(f"Test Asymmetric Score: {score:.2f}")
    
    # Show a few examples
    print("\nSample Predictions:")
    for i in range(5):
        print(f"Engine {i+1}: True RUL = {y_test_clipped[i]}, Predicted RUL = {preds[i]:.1f} (Error: {preds[i]-y_test_clipped[i]:.1f})")

if __name__ == '__main__':
    evaluate_model()