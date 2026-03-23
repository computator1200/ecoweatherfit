# Contributing to EcoWeatherFit

Thank you for your interest in contributing to EcoWeatherFit! This document provides guidelines and standards for contributing to the project.

---

## Development Setup

### Prerequisites
- Python 3.10+
- Git
- API keys (optional — synthetic fallbacks available)

### Getting Started

```bash
# 1. Clone the repository
git clone <repository-url>
cd EcoWeatherFit

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Verify the setup
pytest tests/ -v
```

---

## Project Architecture

The codebase follows a layered architecture with clear separation of concerns:

```
config/          → Configuration (settings, constants, thresholds)
data/            → Data acquisition, preprocessing, feature engineering, splitting
models/          → ML/DL models (Random Forest, LSTM)
recommendations/ → Hybrid recommendation engine (Heuristic + Gemini + Sustainability)
ui/              → Streamlit dashboard components
tests/           → Automated test suite
app.py           → Main application entry point
```

### Design Principles
- **Single Responsibility**: Each module has one clear purpose
- **No Data Leakage**: Scalers fit on training data only; strict chronological splits
- **Graceful Degradation**: Every external dependency has a fallback (synthetic data, heuristic-only mode)
- **Sustainability First**: The Circular Fashion Loop is embedded in every recommendation path

---

## Code Standards

### Style
- Follow PEP 8 with a maximum line length of 100 characters
- Use type hints for function parameters and return values
- Include docstrings for all public functions (NumPy-style)
- Organise imports: stdlib → third-party → local (separated by blank lines)

### Configuration
- All tuneable parameters belong in `config/settings.py`
- API keys are loaded from environment variables — **never hard-coded**
- File paths use `pathlib.Path` for cross-platform compatibility

### Testing
- Write tests for new functionality in the `tests/` directory
- Tests should be self-documenting (clear names, VERIFY comments)
- Run the full test suite before submitting changes: `pytest tests/ -v`

---

## Adding a New Feature

### New Weather Data Source
1. Add the API integration to `data/acquisition.py`
2. Include a synthetic fallback generator
3. Ensure output matches the standard schema (DatetimeIndex named "time", standard columns)
4. Add unit tests in `tests/test_data_pipeline.py`

### New Forecasting Model
1. Create a new module in `models/` following the pattern of `random_forest.py` or `lstm.py`
2. Implement: `train()`, `predict()`, `evaluate()`, `save()`, `load()`, `forecast_7day()`
3. Register the model in `models/__init__.py`
4. Integrate into the `train_models()` function in `app.py`

### New Heuristic Rule
1. Add the rule to `recommendations/heuristic_engine.py` in the `_analyse_day()` method
2. Define the threshold in `config/settings.py` under `HEURISTIC_THRESHOLDS`
3. Update the `DailyConstraints` dataclass if new fields are needed
4. Verify the rule triggers correctly with a unit test

### New Sustainability Content
1. Add content to the appropriate dict in `recommendations/sustainability.py`
2. Ensure new temperature categories are covered
3. Update the Circular Fashion Loop display if adding new steps

---

## Commit Guidelines

- Write clear, descriptive commit messages
- Reference related issues where applicable
- Keep commits focused on a single logical change
- Run tests before committing

---

## Reporting Issues

When reporting a bug:
1. Describe what you expected vs. what happened
2. Include the Python version (`python --version`)
3. Include the error traceback if applicable
4. Note whether you are using real API keys or synthetic mode

---

## Academic Context

This project is developed as part of a final-year Computer Science dissertation focused on combating fast fashion through technology. Contributions should respect:

- **Academic integrity**: All code must be original or properly attributed
- **Transparency**: No silent data manipulation or undocumented preprocessing steps
- **Sustainability**: Every recommendation path must uphold the Circular Fashion Loop
- **Safety**: The heuristic guardrail layer must never be bypassed — GenAI output is always validated

---

## Questions?

For questions about the project architecture or contribution process, please open a GitHub issue.
