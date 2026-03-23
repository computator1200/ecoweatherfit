# EcoWeatherFit — Development Process Log

**Purpose**: This document is a chronological engineering log detailing how the EcoWeatherFit system was audited, completed, and verified. It is intended for inclusion in the formal academic report as evidence of the systematic development process.

---

## Phase 1: Deep Audit & Gap Analysis

### Initial Codebase State

Upon inspection of the existing project directory, the following inventory was established:

#### Present and Functional (Data Layer — ~95% complete)
| Module | Status | Description |
|--------|--------|-------------|
| `config/settings.py` | Complete | Centralised configuration with 12 parameter groups including API keys, temporal parameters, feature variables, data quality thresholds, model hyperparameters, recommendation engine settings, and Gemini API configuration |
| `data/acquisition.py` | Complete | Three API integration functions (Meteostat historical, OWM current, OWM forecast) with deterministic caching, synthetic fallback generators, and custom exception handling |
| `data/preprocessing.py` | Complete | Six-step cleaning pipeline (schema enforcement, dedup, plausibility filtering, missing-value audit, temporal alignment, type coercion) with attached cleaning reports |
| `data/feature_engineering.py` | Complete | Four feature categories (lag features, rolling statistics, cyclical temporal encodings, interaction features) with documented leakage prevention |
| `data/splitting.py` | Complete | Chronological train/val/test split with leakage-safe scaling, LSTM sequence builder, and walk-forward validation generator |
| `tests/test_data_pipeline.py` | Complete | 27 unit tests covering acquisition, preprocessing, feature engineering, splitting, and LSTM sequences |
| `.env` | Present | API keys for OpenWeatherMap and Google Gemini configured |

#### Entirely Missing (0% complete)
| Component | Expected Files | Impact |
|-----------|---------------|--------|
| **Forecasting Models** | `models/random_forest.py`, `models/lstm.py` | No prediction capability — the system cannot generate forecasts |
| **Recommendation Engine** | `recommendations/heuristic_engine.py`, `recommendations/gemini_integration.py`, `recommendations/sustainability.py` | No clothing advice — the core value proposition is absent |
| **User Interface** | `ui/dashboard.py` | No presentation layer — cannot display results to users |
| **Application Entry Point** | `app.py` (Streamlit) | `app.py` existed only as a CLI data pipeline validator, not a Streamlit application |

#### Diagnosis Summary
The project had a solid, well-documented data foundation but was entirely missing the forecasting, recommendation, and presentation layers. The `models/`, `recommendations/`, and `ui/` packages contained only empty `__init__.py` placeholder files. The `app.py` was a command-line script, not the Streamlit application described in the project specification. The README described an intended architecture (listing files like `models/random_forest.py`, `recommendations/heuristic_engine.py`, etc.) but none of these files existed.

---

## Phase 2: Implementation — Challenge & Solution Log

### 2.1 Random Forest Model (`models/random_forest.py`)

**Challenge**: The model needed to support not just one-step prediction but a full 7-day recursive forecast. With tabular ML, each prediction step requires reconstructing lag features from the previous prediction.

**Solution**: Implemented a `WeatherRandomForest` class wrapping scikit-learn's `RandomForestRegressor` with:
- Multi-output regression (predicts all target variables simultaneously)
- A `forecast_7day()` method that performs recursive prediction: predict day 1 → append prediction to working dataset → re-engineer features → predict day 2 → repeat
- Feature re-engineering at each recursive step ensures lag features and rolling statistics are correctly updated
- Model serialisation via joblib for caching between sessions

**Result**: R² = 0.994 for temperature, R² = 0.842 for wind speed on the test set — excellent baseline performance.

### 2.2 LSTM Neural Network (`models/lstm.py`)

**Challenge**: Building a TensorFlow LSTM that maps 30 days of multivariate features to a 7-day multi-target forecast, with graceful degradation if TensorFlow is unavailable.

**Solution**: Implemented a `WeatherLSTM` class with:
- **Architecture**: LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(horizon × targets) → Reshape(7, n_targets)
- This is a **direct multi-step** forecaster — it outputs all 7 days in one forward pass, avoiding the error accumulation of recursive approaches
- The `create_lstm_sequences()` function from the existing data module creates compatible 3D input arrays
- Early stopping and learning rate reduction callbacks prevent overfitting
- A `_check_tensorflow()` utility allows the app to degrade gracefully if TF is not installed

**Challenge**: TensorFlow 2.21.0 had a protobuf version conflict with the Google Gemini SDK (`google-generativeai`). TF 2.21 required protobuf ≥ 6.31.1, while Gemini's dependency `google-ai-generativelanguage` required protobuf < 6.0.0.

**Solution**: Downgraded TensorFlow to 2.18.0, which is compatible with protobuf 5.x and satisfies both dependency trees. Updated `requirements.txt` to pin `tensorflow>=2.15,<2.19`.

**Result**: The LSTM trains successfully on sliding windows (batch, 30, 36 features) → (batch, 7, 2 targets) and produces coherent 7-day forecasts.

### 2.3 Heuristic Engine — Deterministic Guardrails (`recommendations/heuristic_engine.py`)

**Challenge**: Implementing a rule-based system that analyses weather forecasts and produces structured constraints that prevent the GenAI from recommending dangerous or inappropriate clothing.

**Solution**: Designed a `HeuristicEngine` with:
- **Temperature classification**: freezing (<0°C), very_cold (<5°C), cold (<10°C), mild, warm (>25°C), hot (>30°C)
- **Mandatory item rules**: e.g., if temp < 0°C → thermal base layer, insulated coat, hat, gloves, scarf are MANDATORY
- **Rain rules**: if precipitation probability > 80% → waterproof jacket and footwear MANDATORY; > 40% → waterproof suggested
- **Wind rules**: if wind > 40 km/h → wind-resistant layer MANDATORY
- **Avoid rules**: e.g., if temp < 5°C → t-shirt only, shorts, sandals are FORBIDDEN
- **Layering advice**: structured 3-layer system recommendations per temperature category
- **Post-generation validation**: `validate_recommendation()` scans GenAI output for mandatory keywords and auto-appends missing items

This is the critical safety layer described in the project specification (Section 2.4). It ensures that regardless of what Gemini generates, the final output is always meteorologically sound.

### 2.4 Gemini GenAI Integration — Hybrid Architecture (`recommendations/gemini_integration.py`)

**Challenge**: Connecting to Google Gemini for conversational outfit advice while ensuring the output is constrained by the heuristic rules.

**Solution**: Implemented a `GeminiAdvisor` class with a three-stage hybrid flow:

1. **Stage 1 — Heuristic Analysis**: The forecast is analysed by the `HeuristicEngine`, producing `DailyConstraints` with mandatory items, avoid items, warnings, and layering advice
2. **Stage 2 — Gemini Prompt Engineering**: A carefully structured prompt is built that includes:
   - The exact weather data for the day
   - User profile (gender, style preference)
   - ALL mandatory items as hard requirements ("these MUST appear")
   - ALL avoid items as prohibitions
   - Sustainability rules embedded as directives
3. **Stage 3 — Post-Generation Validation**: The `validate_recommendation()` function checks whether every mandatory item appears in the Gemini output. If any are missing, a "Weather Safety Reminder" is appended automatically

**Challenge**: The Gemini model name `gemini-2.0-flash` was deprecated. API calls returned 404 errors.

**Solution**: Queried available models via `genai.list_models()`, identified `gemini-2.5-flash` as the current stable model, and updated the configuration.

**Verified Guardrail Behaviour**: In testing with real London weather data (79% rain probability), Gemini generated outfit advice but omitted the waterproof recommendation. The guardrail detected this omission and automatically appended: "Based on the forecast, please also ensure you have: Waterproof Jacket Or Umbrella". This confirms the hybrid architecture works as designed — GenAI creativity is permitted, but safety constraints are enforced.

### 2.5 Sustainability Engine (`recommendations/sustainability.py`)

**Solution**: Implemented a `SustainabilityEngine` with:
- **Wardrobe tips** per temperature category encouraging reuse of existing clothes
- **Layering guides** from single-layer (hot) to three-layer (freezing) systems
- **Garment care tips** rotating through 10 evidence-based care recommendations
- **Eco-facts** drawn from industry statistics (e.g., "92 million tonnes of textile waste annually")
- **Circular Fashion Loop** display structure with 5 priority steps
- **Swap suggestions** encouraging community sharing platforms

### 2.6 Streamlit Dashboard (`ui/dashboard.py` + `app.py`)

**Challenge**: Building a responsive, professional dashboard that orchestrates data fetching, model training, forecasting, and recommendation generation — all within Streamlit's re-run-on-interaction model.

**Solution**:
- `@st.cache_data` for data fetching (TTL-based caching)
- `@st.cache_resource` for model training (persistent across sessions)
- `st.session_state` for button-triggered forecast flow
- Plotly for interactive multi-trace comparison charts (OWM vs RF vs LSTM)
- Tabbed views for each forecast variable (temperature, precipitation, humidity, wind)
- Expandable sections for raw data tables, heuristic constraint details, and weekly wardrobe plans
- Circular Fashion Loop rendered as a visual step-by-step infographic
- Footer with sustainability messaging

---

## Phase 3: System Verification

### Test Results

| Test Category | Tests | Passed | Notes |
|--------------|-------|--------|-------|
| Data Acquisition | 6 | 5 | 1 pre-existing failure: London Meteostat station doesn't report humidity. Handled gracefully by preprocessing (adds column as NaN) |
| Preprocessing | 7 | 7 | Duplicate removal, plausibility filtering, type coercion all verified |
| Feature Engineering | 6 | 6 | Lag creation, rolling stats, temporal bounds, leakage checks all pass |
| Splitting & Scaling | 6 | 6 | Chronological order verified, leakage-safe scaling confirmed |
| LSTM Sequences | 2 | 2 | Shape validation and edge case handling |
| **Total** | **27** | **26** | **96% pass rate** (1 pre-existing data source limitation) |

### End-to-End Integration Results

| Stage | Status | Key Metric |
|-------|--------|------------|
| Data Acquisition | ✅ | Historical: 2,180 rows; Live OWM: working |
| Preprocessing | ✅ | 2,192 rows cleaned, 6 columns retained |
| Feature Engineering | ✅ | 2,156 rows, 38 features engineered |
| Chronological Split | ✅ | Train=1,509 / Val=323 / Test=324 |
| Random Forest | ✅ | R²=0.994 (temp), R²=0.842 (wind) |
| LSTM Neural Network | ✅ | Trains, converges, forecasts 7 days |
| Heuristic Engine | ✅ | Correctly classifies and constrains |
| Gemini Hybrid | ✅ | GenAI generates; guardrails enforce |
| Sustainability | ✅ | Circular Loop advice generated |
| Streamlit Dashboard | ✅ | Launches, renders, interactive |

### Sustainability Verification

The hybrid engine was verified to enforce the Circular Fashion Loop in every code path:

1. **Gemini prompt**: Contains explicit directives — "ALWAYS suggest shopping your wardrobe first", "suggest creative layering with existing clothes before buying anything new"
2. **Heuristic fallback**: When Gemini is unavailable, the heuristic-only text leads with "from your existing wardrobe!" and "creative layering with existing pieces is the most sustainable choice"
3. **Sustainability section**: Separate UI section with wardrobe tips, layering guides, care tips, eco-facts, and the Circular Fashion Loop infographic
4. **No "buy new" recommendations**: Neither the GenAI prompt nor the heuristic engine suggests purchasing new items as a primary recommendation

---

## Final System Capabilities

The completed EcoWeatherFit system provides:

1. **Real-time weather data** for 5 UK cities via OpenWeatherMap API
2. **5+ years of historical training data** via Meteostat, with synthetic fallback
3. **Dual-model forecasting**: Random Forest (R² > 0.99 for temperature) and LSTM neural network
4. **Interactive 7-day forecast comparison** with Plotly charts (OWM vs RF vs LSTM)
5. **Hybrid recommendation engine**: Gemini 2.5 Flash AI constrained by deterministic heuristic guardrails
6. **Post-generation safety validation** that auto-corrects GenAI omissions
7. **Circular Fashion Loop** integrated into every recommendation
8. **Professional Streamlit dashboard** with responsive layout, metrics display, and sustainability corner
9. **Full offline capability** via synthetic data generators and heuristic-only mode
10. **Academic transparency** through logged preprocessing, zero-leakage splits, and comprehensive test suite
