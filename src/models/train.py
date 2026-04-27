import polars as pl
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import sys

# Ensure src module can be imported from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocess import load_and_clean_data, add_rul, add_rolling_features

def asymmetric_loss(y_true, y_pred):
    """
    NASA's Asymmetric scoring metric.
    Overestimating RUL causes catastrophic failure (penalized heavily: e^(error/10)).
    Underestimating causes early maintenance (e^(-error/13)).
    """
    diff = y_pred - y_true
    penalty = np.where(diff > 0, np.exp(diff / 10.0) - 1, np.exp(-diff / 13.0) - 1)
    return np.sum(penalty)

def train_model():
    print("Loading and preprocessing training data...")
    train_df = load_and_clean_data('datasets/RUL/train_FD001.txt')
    train_df = add_rul(train_df)
    train_df = add_rolling_features(train_df, window_size=15)
    
    # Drop rows with nulls created by rolling windows (the first 14 cycles of each engine)
    train_df = train_df.drop_nulls()

    # Define our features based on our EDA
    features = [
        'cycle', 
        'sensor_4', 'sensor_11', 
        'sensor_4_rolling_mean', 'sensor_11_rolling_mean', 
        'sensor_4_rolling_std', 'sensor_11_rolling_std'
    ]
    target = 'RUL'

    # Convert to Pandas/NumPy for LightGBM
    X = train_df.select(features).to_pandas()
    y = train_df.select(target).to_pandas().values.ravel()
    
    # Standard Industry Practice for CMAPSS: Piecewise Linear RUL
    # We clip the maximum RUL to 130 during training
    MAX_RUL = 130
    y = np.clip(y, a_min=None, a_max=MAX_RUL)

    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    score = asymmetric_loss(y, preds)
    
    print(f"Training RMSE: {rmse:.2f}")
    print(f"Training Asymmetric Score: {score:.2f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lgb_model.txt'
    model.booster_.save_model(model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == '__main__':
    train_model()