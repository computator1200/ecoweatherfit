# EcoWeatherFit

**A Sustainable Weather Forecasting and Clothing Recommendation System**

## Overview

EcoWeatherFit is a unified digital platform that predicts 7-day weather conditions using machine learning and deep learning, and translates those forecasts into practical, personalised, and sustainability-aware clothing guidance. The system addresses the highly changeable UK weather and the environmental impact of the fast fashion industry.

## Project Structure

```
EcoWeatherFit/
├── config/
│   └── settings.py              # Centralised configuration & constants
├── data/
│   ├── __init__.py              # Package exports
│   ├── acquisition.py           # API integrations (Meteostat, OWM)
│   ├── preprocessing.py         # Cleaning, alignment, plausibility filtering
│   ├── feature_engineering.py   # Lags, rolling stats, cyclical encodings
│   └── splitting.py             # Chronological splits & leakage-safe scaling
├── models/
│   ├── baselines.py             # Persistence & seasonal naive baselines
│   ├── random_forest.py         # RF with lagged features
│   └── lstm.py                  # LSTM sliding-window forecaster
├── recommendations/
│   ├── heuristic_engine.py      # Rule-based safety guardrails
│   ├── gemini_integration.py    # Gemini GenAI conversational advice
│   └── sustainability.py        # Digital nudging & eco-aware suggestions
├── ui/
│   └── dashboard.py             # Streamlit responsive dashboard
├── tests/
│   └── test_data_pipeline.py    # Comprehensive pipeline unit tests
├── cache/                       # API response cache (git-ignored)
├── logs/                        # Runtime logs (git-ignored)
├── requirements.txt
├── app.py                       # Main Streamlit entry point
└── README.md
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML / DL | TensorFlow/Keras (LSTM), Scikit-learn (Random Forest) |
| Data | Pandas, NumPy, Meteostat |
| Visualisation | Matplotlib, Plotly |
| APIs | OpenWeatherMap, Meteostat, Google Gemini |
| UI | Streamlit |

## Setup

```bash
# 1. Clone and enter the project
cd EcoWeatherFit

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables (or create a .env file)
set OWM_API_KEY=your_openweathermap_key
set GEMINI_API_KEY=your_gemini_api_key

# 5. Run tests
pytest tests/ -v

# 6. Launch the dashboard
streamlit run app.py
```

## Data Pipeline (Step 1)

### Data Acquisition
- **Meteostat**: Bulk historical daily observations for 5+ years across UK cities
- **OpenWeatherMap**: Real-time current conditions and 7-day forecasts
- **Synthetic fallback**: Development mode generates realistic UK weather patterns when API keys are unavailable

### Preprocessing
1. Schema enforcement — expected columns guaranteed
2. Duplicate removal — exact timestamp deduplication
3. Plausibility filtering — domain-bound validation (e.g., temp ∈ [-30, 45] °C)
4. Missing-value audit — quantified, logged, conditionally interpolated
5. Temporal alignment — strict daily frequency, limited forward-fill
6. Type coercion — float32 for memory efficiency

### Feature Engineering
- **Lag features**: 1, 2, 3, 5, 7, 14-day lags for all target variables
- **Rolling statistics**: 3, 7, 14-day moving averages and standard deviations
- **Cyclical encodings**: sin/cos transforms of day-of-year, month, day-of-week
- **Interaction features**: wind chill, heat index, is_rainy flag

### Data Splitting (Zero Leakage)
- Strict chronological split: 70% train / 15% validation / 15% test
- Scalers fit on training data only
- Optional gap days between splits to mitigate autocorrelation
- Walk-forward (rolling-origin) validation generator included
- LSTM sliding-window sequence builder (lookback=30, horizon=7)

## Academic Integrity

- **No data leakage**: All splits are strictly temporal; scalers are trained-only
- **Full transparency**: Every cleaning decision is logged and reported
- **Reproducibility**: All random seeds are fixed; configuration is centralised
- **Synthetic data is labelled**: Development fallbacks are never passed off as real

## License

This project is developed for academic purposes as part of a final-year Computer Science dissertation.
