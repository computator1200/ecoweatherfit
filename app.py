"""
EcoWeatherFit — Main Streamlit Application Entry Point
=======================================================
Launch with:  streamlit run app.py

This file will be fully implemented in Step 4 (UI Dashboard).
For now it serves as a placeholder that validates the data pipeline.
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from config.settings import LOG_FORMAT, LOG_LEVEL

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("EcoWeatherFit")


def main():
    """Quick validation of the data pipeline (pre-UI)."""
    logger.info("=" * 60)
    logger.info("  EcoWeatherFit — Data Pipeline Validation")
    logger.info("=" * 60)

    # 1. Acquire data (synthetic fallback if no API key)
    from data.acquisition import fetch_meteostat_historical
    raw = fetch_meteostat_historical(city="London", use_cache=True)
    logger.info("Raw data shape: %s", raw.shape)

    # 2. Preprocess
    from data.preprocessing import preprocess_pipeline, quality_summary
    cleaned = preprocess_pipeline(raw)
    logger.info("Cleaned data shape: %s", cleaned.shape)

    summary = quality_summary(cleaned)
    print("\n📊 Data Quality Summary:\n")
    print(summary.to_string())

    # 3. Feature engineering
    from data.feature_engineering import engineer_features, get_feature_names
    featured = engineer_features(cleaned, drop_na_rows=True)
    feature_names = get_feature_names(featured)
    logger.info("Engineered features: %d columns, %d rows", len(featured.columns), len(featured))
    logger.info("Feature names (%d): %s", len(feature_names), feature_names[:10])

    # 4. Split
    from data.splitting import time_based_split
    split = time_based_split(featured, scale=True)
    logger.info("Split complete — Train: %d, Val: %d, Test: %d",
                split.metadata["n_train"], split.metadata["n_val"], split.metadata["n_test"])

    # 5. LSTM sequences
    from data.splitting import create_lstm_sequences
    X_seq, y_seq = create_lstm_sequences(split.X_train, split.y_train)
    logger.info("LSTM sequences — X: %s, y: %s", X_seq.shape, y_seq.shape)

    logger.info("=" * 60)
    logger.info("  ✅ Pipeline validation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
