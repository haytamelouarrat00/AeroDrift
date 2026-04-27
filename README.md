# ✈️ AeroDrift: Real-Time Predictive Maintenance Engine

A production-grade, end-to-end Machine Learning system designed to predict the Remaining Useful Life (RUL) of turbofan engines using high-frequency IoT telemetry data. 

Built with **Polars**, **LightGBM**, and **FastAPI**, this project demonstrates how to transition from raw, noisy sensor data to a stateful, real-time inference microservice.

---

## 🚀 The Business Problem
In aerospace and heavy industry, maintenance timing is critical:
* **Overestimating RUL** (predicting the engine is healthy when it's not) leads to catastrophic mid-flight failures.
* **Underestimating RUL** (predicting failure too early) costs airlines millions in premature maintenance and grounded flights.

**AeroDrift** solves this by utilizing **NASA's CMAPSS dataset** to train a gradient-boosting model optimized specifically for an **Asymmetric Loss Function**—heavily penalizing dangerous overestimations while maintaining tight accuracy as the engine approaches failure.

---

## 🧠 Key Engineering Highlights

### 1. High-Performance Data Pipeline (Polars)
Instead of standard Pandas, the data pipeline leverages **Polars** (written in Rust) to compute time-series features (Rolling Means and Rolling Standard Deviations) across multiple engines in parallel, capturing the *rate of sensor drift* rather than just absolute values.

### 2. Domain-Specific Modeling (Piecewise Linear RUL)
During the early life of a turbofan engine, degradation is practically unobservable (sensors remain flat). To prevent the model from wildly guessing the RUL of healthy engines, the training targets were clipped to a **Piecewise Linear RUL maximum of 130 cycles**. 
* *Result:* The model correctly outputs "Healthy" for new engines and only begins aggressively tracking RUL once actual physical degradation begins.

### 3. Stateful Real-Time Inference API (FastAPI)
The prediction system isn't just a static script; it's a **stateful FastAPI microservice**. 
* Edge devices (IoT sensors) stream telemetry row-by-row via HTTP.
* The API maintains an in-memory double-ended queue (`collections.deque`) for every active engine in the fleet.
* Once the API buffers enough context (15 cycles), it dynamically computes the rolling features on-the-fly, runs LightGBM inference, and returns real-time alerts (🟢 HEALTHY, 🟡 WARNING, 🔴 CRITICAL).

---

## 🛠️ Tech Stack
* **Data Processing:** Polars, NumPy
* **Machine Learning:** LightGBM, Scikit-Learn
* **Model Serving:** FastAPI, Uvicorn, Pydantic
* **Simulation:** Python `requests`

---

## 📂 Project Structure

```text
AERODRIFT/
├── datasets/RUL/          # Raw NASA CMAPSS telemetry data
├── models/                # Compiled model binaries (lgb_model.txt)
├── src/
│   ├── data/
│   │   └── preprocess.py  # Polars data cleaning and feature engineering
│   ├── models/
│   │   ├── train.py       # Model training and asymmetric loss optimization
│   │   └── evaluate.py    # Test set evaluation script
│   └── api/
│       ├── main.py        # Stateful FastAPI inference server
│       └── simulator.py   # IoT telemetry streaming simulator
├── aerodrift.ipynb        # Initial Exploratory Data Analysis (EDA)
└── README.md
```

---

## 💻 How to Run Locally

### 1. Start the Inference Server
Open a terminal and start the stateful FastAPI server:
```bash
uvicorn src.api.main:app --reload
```
*The API docs will be available at `http://127.0.0.1:8000/docs`*

### 2. Run the Telemetry Simulator
Open a **second terminal** and simulate an engine streaming live IoT data to the server:
```bash
# Simulate Engine #3 sending data every 0.1 seconds
python src/api/simulator.py --engine 3 --delay 0.1
```

### 3. Watch the Predictions
You will see the simulator buffer the initial data, report `🟢 HEALTHY`, and eventually trigger `🟡 WARNING` and `🔴 CRITICAL` alerts as the engine naturally degrades in real-time!