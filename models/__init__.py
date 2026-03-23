# EcoWeatherFit - Models Package
"""
models/
=======
Weather forecasting models — classical ML and deep learning.

Modules
-------
random_forest  – Multi-output Random Forest (baseline)
lstm           – Sliding-window LSTM deep learning forecaster
"""

from models.random_forest import WeatherRandomForest
from models.lstm import WeatherLSTM
