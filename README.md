# EcoWeatherFit

**A Sustainable Weather Forecasting and Clothing Recommendation System**

EcoWeatherFit is a unified digital platform that predicts 7-day weather conditions using machine learning (Random Forest) and deep learning (LSTM), and translates those forecasts into practical, personalised, sustainability-aware clothing guidance. The system addresses the highly changeable UK weather and the environmental impact of the fast fashion industry by promoting **circular fashion** principles: reuse, layering, repair, and responsible purchasing.

---

## Key Features

- **Dual-Model Forecasting**: Random Forest (classical ML baseline) and LSTM neural network (deep learning) trained on historical Meteostat data
- **Live Weather Integration**: Real-time data from OpenWeatherMap API for current conditions and 5-day forecasts
- **Hybrid Recommendation Engine**: Gemini AI generates conversational outfit advice, constrained by deterministic heuristic guardrails to prevent hallucinations
- **Circular Fashion Loop**: Every recommendation prioritises reusing existing wardrobe items over buying new
- **Interactive Dashboard**: Professional Streamlit UI with Plotly charts, forecast comparisons, and sustainability tips
- **Academic Transparency**: Zero data leakage, logged preprocessing, reproducible splits

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                       │
│  Current Weather │ 7-Day Forecast Charts │ Recommendations  │
├─────────────────┬───────────────────────┬───────────────────┤
│   DATA LAYER    │   FORECASTING LAYER   │  RECOMMENDATION   │
│                 │                       │      ENGINE        │
│  Meteostat API  │  Random Forest Model  │                   │
│  (Historical)   │  (Baseline: R²=0.99)  │  Gemini GenAI     │
│                 │                       │       +            │
│  OpenWeatherMap │  LSTM Neural Network  │  Heuristic Rules   │
│  (Live/Forecast)│  (30-day → 7-day)     │  (Safety Guards)   │
│                 │                       │       +            │
│  Preprocessing  │  Evaluation Metrics   │  Sustainability    │
│  & Feature Eng. │  (MAE, RMSE, R²)      │  (Circular Loop)   │
└─────────────────┴───────────────────────┴───────────────────┘
```

### Hybrid Recommendation Flow

```
Forecast Data ──→ Heuristic Engine ──→ Deterministic Constraints
                                            │
                                            ▼
                                     Gemini GenAI Prompt
                                     (constraints embedded)
                                            │
                                            ▼
                                     GenAI Output Text
                                            │
                                            ▼
                                     Post-Generation Validation
                                     (guardrails check mandatory items)
                                            │
                                            ▼
                                     Final Recommendation
                                     + Sustainability Nudges
```

---

## Project Structure

```
EcoWeatherFit/
├── config/
│   └── settings.py                 # Centralised configuration & constants
├── data/
│   ├── __init__.py                 # Package exports
│   ├── acquisition.py              # API integrations (Meteostat, OWM)
│   ├── preprocessing.py            # Cleaning, alignment, plausibility filtering
│   ├── feature_engineering.py      # Lags, rolling stats, cyclical encodings
│   └── splitting.py                # Chronological splits & leakage-safe scaling
├── models/
│   ├── __init__.py                 # Package exports
│   ├── random_forest.py            # Multi-output Random Forest forecaster
│   └── lstm.py                     # Sliding-window LSTM deep learning model
├── recommendations/
│   ├── __init__.py                 # Package exports
│   ├── heuristic_engine.py         # Deterministic safety guardrails
│   ├── gemini_integration.py       # Gemini GenAI hybrid architecture
│   └── sustainability.py           # Circular Fashion Loop & eco-advice
├── ui/
│   ├── __init__.py                 # Package exports
│   └── dashboard.py                # Streamlit UI components
├── tests/
│   └── test_data_pipeline.py       # 27 unit tests for data pipeline
├── cache/                          # API response cache (git-ignored)
├── logs/                           # Runtime logs (git-ignored)
├── models/saved/                   # Trained model artifacts (git-ignored)
├── requirements.txt                # Python dependencies
├── app.py                          # Main Streamlit entry point
├── .env.example                    # Environment variable template
└── README.md
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | Core runtime |
| ML (Classical) | Scikit-learn | Random Forest multi-output regression |
| DL (Neural) | TensorFlow/Keras 2.18 | LSTM sliding-window forecaster |
| Data | Pandas, NumPy | Data manipulation & numerical computation |
| Weather APIs | Meteostat, OpenWeatherMap | Historical & live weather data |
| GenAI | Google Gemini 2.5 Flash | Conversational outfit recommendations |
| Visualisation | Plotly | Interactive forecast comparison charts |
| UI Framework | Streamlit | Responsive web dashboard |
| Testing | pytest | Automated unit test suite |

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- API keys for OpenWeatherMap and Google Gemini (optional — system works with synthetic data)

### Steps

```bash
# 1. Clone and enter the project
git clone <repository-url>
cd EcoWeatherFit

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your API keys:
#   OWM_API_KEY=your_openweathermap_key
#   GEMINI_API_KEY=your_gemini_api_key

# 5. Run the test suite
pytest tests/ -v

# 6. Launch the dashboard
streamlit run app.py
```

The application will open at `http://localhost:8501`.

### Running Without API Keys

EcoWeatherFit includes synthetic data generators for all external APIs. If no API keys are configured:
- **Meteostat**: Falls back to realistic sinusoidal UK weather patterns with Gaussian noise
- **OpenWeatherMap**: Generates synthetic current conditions and 7-day forecasts
- **Gemini**: Uses the heuristic-only recommendation engine (rule-based, no GenAI)

This ensures the full system can be demonstrated and tested offline.

---

## Usage

1. **Select a City**: Choose from London, Manchester, Edinburgh, Birmingham, or Cardiff
2. **Set Your Profile**: Select gender presentation and style preference
3. **Get Forecast**: Click "Get Forecast & Advice" to trigger the full pipeline
4. **View Results**:
   - Current weather conditions with key metrics
   - 7-day forecast comparison (OWM API vs Random Forest vs LSTM)
   - Model performance metrics (MAE, RMSE, R²)
   - Personalised outfit recommendation with sustainability tips
   - Circular Fashion Loop guidance

---

## Data Pipeline

### Acquisition
- **Meteostat**: 5+ years of daily historical observations via the `meteostat` Python library
- **OpenWeatherMap**: Real-time conditions (Current Weather API) and 5-day/3-hour forecasts (aggregated to daily)
- **Caching**: All API responses cached locally with configurable TTL to respect rate limits

### Preprocessing (6 Steps)
1. Schema enforcement — expected columns guaranteed (missing columns added as NaN)
2. Duplicate removal — exact timestamp deduplication
3. Plausibility filtering — domain-bound validation (e.g., temp in [-30, 45] °C)
4. Missing-value audit — quantified, logged, conditionally dropped (>15% threshold)
5. Temporal alignment — strict daily frequency with limited forward-fill (max 3 days)
6. Type coercion — float32 for memory efficiency and tensor compatibility

### Feature Engineering
- **Lag features**: 1, 2, 3, 5, 7, 14-day lags for all target variables
- **Rolling statistics**: 3, 7, 14-day moving averages and standard deviations
- **Cyclical encodings**: sin/cos transforms of day-of-year, month, day-of-week
- **Interaction features**: wind chill index, heat index, is_rainy flag, temperature range

### Data Splitting (Zero Leakage)
- Strict chronological split: 70% train / 15% validation / 15% test
- **Scalers fit on training data only** — no future information leakage
- Optional gap days between splits to mitigate autocorrelation
- LSTM sliding-window sequence builder (lookback=30, horizon=7)

---

## Models

### Random Forest
- Multi-output regressor (one model predicts all target variables)
- 300 trees, max depth 15, min samples leaf 5
- Recursive 7-day forecasting (one-step prediction fed back as lag features)
- Typical performance: **R² > 0.99 for temperature**

### LSTM Neural Network
- Encoder architecture: LSTM(64) → Dropout(0.2) → Dense(32) → Dense(horizon × targets)
- Direct multi-step forecaster: inputs 30 days, outputs 7 days simultaneously
- Trained with early stopping and learning rate reduction
- Adam optimiser with MSE loss

---

## Hybrid Recommendation Engine

The recommendation engine uses a **Hybrid Architecture** combining GenAI creativity with deterministic safety:

### Heuristic Guardrails (Layer 1)
Deterministic rules based on weather thresholds:
- **Freezing (<0°C)**: Thermal base layer, insulated coat, hat, gloves, scarf mandatory
- **Very Cold (<5°C)**: Thermal layers essential
- **Rain >80%**: Waterproof jacket and footwear mandatory
- **Rain >40%**: Waterproof jacket or umbrella suggested
- **Wind >40 km/h**: Wind-resistant outer layer mandatory

### Gemini GenAI (Layer 2)
- Generates personalised, conversational outfit advice
- Prompt includes all heuristic constraints as hard requirements
- User profile (gender, style) shapes the advice tone

### Post-Generation Validation (Layer 3)
- Checks GenAI output against mandatory constraints
- Any missing items are automatically appended
- Ensures factual grounding regardless of GenAI behaviour

### Circular Fashion Loop
All recommendations prioritise:
1. **Shop Your Wardrobe** — reuse existing items
2. **Layer Creatively** — combine existing pieces
3. **Swap or Borrow** — community sharing
4. **Buy Second-Hand** — charity shops, Vinted, Depop
5. **Buy Sustainable** — eco brands as last resort

---

## Academic Integrity

- **No data leakage**: All splits are strictly temporal; scalers fitted on training data only
- **Full transparency**: Every preprocessing decision is logged with reasons
- **Reproducibility**: Fixed random seeds; centralised configuration
- **Synthetic data labelled**: Development fallbacks are never passed off as real observations
- **Documented architecture**: Hybrid engine guardrails are traceable and verifiable

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_data_pipeline.py::TestSplitting -v
```

The test suite covers:
- Data acquisition (synthetic fallback verification)
- Preprocessing (duplicate removal, plausibility filtering, type coercion)
- Feature engineering (lag creation, temporal encoding bounds, leakage checks)
- Data splitting (chronological order, scaling leakage prevention, size validation)
- LSTM sequence shapes and edge cases

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OWM_API_KEY` | OpenWeatherMap API key | No (synthetic fallback) |
| `GEMINI_API_KEY` | Google Gemini API key | No (heuristic-only fallback) |
| `ECOWEATHERFIT_LOG_LEVEL` | Logging level (INFO, DEBUG, etc.) | No (default: INFO) |

---

## License

This project is developed for academic purposes as part of a final-year Computer Science dissertation.
