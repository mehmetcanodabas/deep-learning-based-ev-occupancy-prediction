# deep-learning-based-ev-occupancy-prediction

Deep Learning-Based EV Charging Point Occupancy Forecasting Using PyTorch LSTM

This project implements a deep learning workflow for 6-hour EV charging point occupancy forecasting using PyTorch and the UrbanEV dataset.

The solution uses a Long Short-Term Memory (LSTM) neural network to predict the number of occupied EV charging points in each zone for the next six hours, based on the previous 48 hours of hourly occupancy data.

A single shared LSTM model is trained across 30 selected charging zones using chronological training, validation, and test splits. The workflow includes data preparation, normalization, sequence generation, CPU-based model training, early stopping, and multi-horizon evaluation.

Final predictions are evaluated against actual occupancy values using MAE, RMSE, and WAPE. A separate six-hour forecasting example demonstrates model predictions across five selected zones.

### Key Results

- Developed a **PyTorch LSTM model for 6-hour EV charging point occupancy forecasting**.
- Selected **30 charging zones with different occupancy levels** from the UrbanEV dataset.
- Trained a **single shared LSTM model across 30 charging zones**, enabling six-hour occupancy predictions for each zone.
- Used **48 hours of historical occupancy data** to predict the next **6 hourly occupancy values**.
- Implemented a **single-layer LSTM with 32 hidden units and 4,678 trainable parameters**.
- Trained the model on CPU using Adam optimization, MSE loss, and early stopping.
- Achieved **10.71% validation WAPE** and **10.74% test WAPE**.
- Demonstrated a **six-hour forecasting case study across five selected zones**, achieving individual WAPE values between **3.73% and 8.37%** for the selected forecast window.

### Result Snapshots

[Five-Zone Forecast Performance](docs/results/five_zone_case_study.png): Comparison of six-hour forecasting accuracy across five selected charging zones on February 15, 2023, including WAPE and average actual versus predicted occupied charging points.

[Five-Zone Actual vs. Predicted Occupancy](docs/results/actual_vs_predicted.png): Hour-by-hour comparison of actual and predicted occupied charging points for five selected zones during a single six-hour forecast (13:00–18:00, February 15, 2023).

**Keywords:** Python, Deep Learning, Energy Data Analytics, EV Charging, LSTM, Machine Learning, Multi-Horizon Forecasting, Neural Networks, PyTorch, Time-Series Forecasting.

**Data Source:** [UrbanEV Dataset — IntelligentSystemsLab](https://github.com/IntelligentSystemsLab/UrbanEV), September 2022 – February 2023.
