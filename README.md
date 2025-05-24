# Hourly Dst Index Forecasting (2010 – 2025)

Forecasting geomagnetic storm intensity 1 to 12 hours ahead using deep-learning models trained on 15 years of NASA OMNI solar-wind data.

## What is the Dst Index?

The Disturbance Storm-Time (Dst) index quantifies the globally averaged deviation of the Earth’s horizontal magnetic field component at low-latitude observatories (in nano Tesla, nT). A strongly negative Dst indicates an enhanced westward ring current encircling the planet — a signature of geomagnetic storms driven by solar-wind energy input. Reliable Dst forecasts are critical for:

- Satellite orbit and attitude control
- Spacecraft radiation-belt dose management
- Ground-induced current (GIC) mitigation for power-grid operators
- High-frequency radio communication & GNSS accuracy

### Storm-Level Classification

| Level        | Dst Range (nT) | Typical Impact                              |
|--------------|----------------|---------------------------------------------|
| Quiet        | > −20          | Nominal magnetosphere                       |
| Weak         | −20 to −50     | Auroral activity, minor GICs                |
| Moderate     | −50 to −100    | Surface charging, single-event upsets       |
| Intense      | −100 to −200   | Radiation-belt enhancements, pipeline corrosion |
| Super-storm  | < −200         | Widespread GICs, transformer damage, satellite drag |

(Based on NOAA SWPC and Kyoto WDC thresholds)

## Dataset

| Item       | Value                              |
|------------|------------------------------------|
| Source     | NASA OMNIWeb 1-hr merged dataset   |
| Period     | 2010 – 2025                        |
| Cadence    | Hourly                             |
| Samples    | ~131,736 rows                      |
| Features   | Solar-wind plasma (V, n, Pdyn), IMF (Bz, B, clock-angle), past Dst lags, etc. |

## Methodology

### Feature Selection
A Random-Forest Regressor ranked 45 candidate drivers; the top 20 with the highest Gini importance were retained.

### Temporal Split
Chronological split → 80% train, 10% validation, 10% hold-out test.

### Models & Hyper-Parameter Tuning

| Model       | Search Space (Example)                              |
|-------------|----------------------------------------------------|
| ANN         | layers ∈ {2,3}, units ∈ {64,128}, dropout ∈ {0.2,0.3} |
| CNN         | filters ∈ {16,32}, kernel ∈ {2,3}, dropout ∈ {0.2,0.3} |
| LSTM        | units ∈ {32,64}, dropout ∈ {0.2,0.3}               |
| Bi-LSTM     | units ∈ {32,64}, dropout ∈ {0.2,0.3}               |
| CNN + LSTM  | filters ∈ {16,32}, kernel ∈ {2,3}, units ∈ {32,64}, dropout ∈ {0.2,0.3} |

Grid/random search with early stopping was executed for 30 epochs (patience = 5) in each case.

## Results

| Model       | Val RMSE | Test RMSE | Notes                              |
|-------------|----------|-----------|------------------------------------|
| ANN         | 0.0031   | 0.0033    | 2 dense layers                     |
| CNN         | 0.0024   | 0.0026    | filters = 32, kernel = 3           |
| LSTM        | 0.0022   | 0.0024    | units = 64                         |
| Bi-LSTM     | 0.0020   | 0.0022    | bidirectional                      |
| CNN + LSTM  | 0.0018   | 0.0021    | filters = 32, kernel = 2, units = 32 |

The hybrid CNN + LSTM architecture outperformed all baselines, capturing both local temporal patterns (via convolution) and long-range dependencies (via recurrent layers).


## Quick-Start

git clone https://github.com/m-vp/DST-Index-Forecasting.git
cd DST-Index-Forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --model cnn_lstm

The notebook `tsa_final_code.ipynb` reproduces the complete pipeline from data ingestion to inference.


## License

This project is released under the MIT License — see the LICENSE file for details.

Developed with ❤️ by Prashanth (Amrita Vishwa Vidyapeetham, Bengaluru) — contributions welcome!

