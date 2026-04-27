import time
import requests
import json
import argparse
import sys
import os

# Ensure src module can be imported from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import polars as pl
from src.data.preprocess import load_and_clean_data

API_URL = "http://localhost:8000/predict"

def run_simulator(engine_id: int, delay: float):
    print(f"Loading test dataset...")
    # Make sure we use the preprocessed loader
    try:
        df = load_and_clean_data('datasets/RUL/test_FD001.txt')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Filter for the specific engine
    engine_data = df.filter(pl.col("engine_id") == engine_id)
    
    if engine_data.height == 0:
        print(f"Error: Engine {engine_id} not found in test data.")
        return
        
    print(f"Starting telemetry simulation for Engine {engine_id}...")
    print(f"Sending {engine_data.height} cycles of data to {API_URL} with a {delay}s delay...\n")
    
    for row in engine_data.iter_rows(named=True):
        payload = {
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"]),
            "sensor_4": float(row["sensor_4"]),
            "sensor_11": float(row["sensor_11"])
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result["status"] == "buffering":
                print(f"[Cycle {result['cycle']:03d}] {result['message']}")
            else:
                rul = result["RUL"]
                if rul < 30:
                    print(f"[Cycle {result['cycle']:03d}] 🔴 CRITICAL: Predicted RUL = {rul:.1f} cycles! Maintenance required.")
                elif rul < 60:
                    print(f"[Cycle {result['cycle']:03d}] 🟡 WARNING: Predicted RUL = {rul:.1f} cycles. Schedule inspection.")
                else:
                    print(f"[Cycle {result['cycle']:03d}] 🟢 HEALTHY: Predicted RUL = {rul:.1f} cycles.")
                    
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to API: {e}")
            print("Make sure the FastAPI server is running: uvicorn src.api.main:app")
            break
            
        time.sleep(delay)
    
    print("\nSimulation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate turbofan telemetry streaming to the API.")
    parser.add_argument("--engine", type=int, default=1, help="Engine ID to simulate (default: 1)")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between HTTP requests in seconds (default: 0.2)")
    args = parser.parse_args()
    
    run_simulator(args.engine, args.delay)
