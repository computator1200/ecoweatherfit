"""
Full back-test: train RF + LSTM on data through 2025-12-01, forecast Dec 2-8,
compare against real observations. Produces 3 publication-ready charts.
"""
import os, sys, logging, warnings
logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.acquisition import fetch_meteostat_historical
from data.preprocessing import preprocess_pipeline
from data.feature_engineering import engineer_features
from data.splitting import time_based_split, create_lstm_sequences
from models.random_forest import WeatherRandomForest
from models.lstm import WeatherLSTM
from config.settings import LSTM_LOOKBACK_WINDOW
import data.feature_engineering as fe_mod

CITY = "London"
HORIZON = 7
HOLDOUT = 30

# 1. Load + truncate history
df = preprocess_pipeline(fetch_meteostat_historical(CITY))
cutoff = df.index.max() - pd.Timedelta(days=HOLDOUT)
df_train_hist = df.loc[:cutoff]
df_actual = df.loc[cutoff + pd.Timedelta(days=1):].iloc[:HORIZON]
fe_train = engineer_features(df_train_hist, drop_na_rows=True)
tgts = [c for c in ["temp", "wind_speed"] if c in fe_train.columns]
split = time_based_split(fe_train, target_cols=tgts)

# 2. Train Random Forest with the persistence-imputation fix
rf = WeatherRandomForest()
rf.train(split.X_train, split.y_train, split.X_val, split.y_val)
fc_rf_post = rf.forecast_7day(
    recent_raw=fe_train.tail(60),
    feature_scaler=split.feature_scaler,
    target_scaler=split.target_scaler,
    feature_cols=split.metadata["feature_cols"],
    target_cols=split.metadata["target_cols"],
)

# 3. Replay the pre-fix RF behaviour (NaN-fill for non-target columns)
def forecast_prefix(rf_model, recent_raw, fs, ts, fcols, tcols, horizon=7):
    working = recent_raw.copy()
    preds = []
    last_date = working.index.max()
    for d in range(1, horizon + 1):
        feat = fe_mod.engineer_features(working, drop_na_rows=False)
        for c in fcols:
            if c not in feat.columns:
                feat[c] = 0.0
        last_row = feat[fcols].iloc[[-1]].fillna(0.0)
        ys = rf_model.predict(fs.transform(last_row.values))
        yi = ts.inverse_transform(ys)
        pred = {c: yi[0, i] for i, c in enumerate(tcols)}
        pred_date = last_date + pd.Timedelta(days=d)
        row = {c: pred.get(c, np.nan) for c in working.columns}
        new = pd.DataFrame(row, index=[pred_date])
        new.index.name = "time"
        working = pd.concat([working, new])
        preds.append(pred)
    return pd.DataFrame(
        preds,
        index=pd.date_range(
            last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        ),
        columns=tcols,
    )

fc_rf_pre = forecast_prefix(
    rf,
    fe_train.tail(60),
    split.feature_scaler,
    split.target_scaler,
    split.metadata["feature_cols"],
    split.metadata["target_cols"],
)

# 4. Train LSTM v2 (multi-head)
lstm = WeatherLSTM()
lstm.build(
    n_features=split.X_train.shape[1],
    n_targets=split.y_train.shape[1],
    target_names=tgts,
)
Xtr, ytr = create_lstm_sequences(split.X_train, split.y_train)
Xva, yva = create_lstm_sequences(split.X_val, split.y_val)
lstm.train(Xtr, ytr, Xva, yva)

recent_lstm = fe_train[split.metadata["feature_cols"]].tail(LSTM_LOOKBACK_WINDOW)
recent_lstm_scaled = split.feature_scaler.transform(recent_lstm.values).astype(np.float32)
fc_lstm = lstm.forecast_7day(
    recent_features_scaled=recent_lstm_scaled,
    target_scaler=split.target_scaler,
    target_cols=split.metadata["target_cols"],
    last_date=fe_train.index.max(),
)

# 5. Plot
dates = df_actual.index

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 110,
})


def style_dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.grid(True, linestyle=":", alpha=0.4)


def col(s, c):
    return s[c].values


# --- Figure 6a: Temperature back-test ---
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(dates, col(df_actual, "temp"), "o-", color="#2196F3",
        linewidth=2.2, label="Observed (Meteostat)")
ax.plot(dates, col(fc_rf_post, "temp"), "s--", color="#FF8800",
        linewidth=1.8, label="Random Forest (post-fix)")
ax.plot(dates, col(fc_lstm, "temp"), "^:", color="#4CAF50",
        linewidth=1.8, label="LSTM v2")
mae_rf = mean_absolute_error(col(df_actual, "temp"), col(fc_rf_post, "temp"))
mae_lstm = mean_absolute_error(col(df_actual, "temp"), col(fc_lstm, "temp"))
ax.set_title(
    f"London — held-out 7-day temperature forecast vs observation\n"
    f"(training ended {cutoff.date()}, forecast window {dates[0].date()} - {dates[-1].date()})",
    fontsize=11,
)
ax.set_ylabel("Temperature (°C)")
style_dates(ax)
ax.legend(loc="upper left", framealpha=0.95)
ax.text(
    0.99, 0.04,
    f"7-day MAE — RF: {mae_rf:.2f} °C   |   LSTM: {mae_lstm:.2f} °C",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"),
)
plt.tight_layout()
plt.savefig("figures/fig6a_temp_backtest.png", bbox_inches="tight", dpi=140)
plt.close()
print("Saved figures/fig6a_temp_backtest.png")

# --- Figure 6b: Wind speed back-test ---
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(dates, col(df_actual, "wind_speed"), "o-", color="#2196F3",
        linewidth=2.2, label="Observed (Meteostat)")
ax.plot(dates, col(fc_rf_post, "wind_speed"), "s--", color="#FF8800",
        linewidth=1.8, label="Random Forest (post-fix)")
ax.plot(dates, col(fc_lstm, "wind_speed"), "^:", color="#4CAF50",
        linewidth=1.8, label="LSTM v2")
mae_rf_w = mean_absolute_error(col(df_actual, "wind_speed"),
                                col(fc_rf_post, "wind_speed"))
mae_lstm_w = mean_absolute_error(col(df_actual, "wind_speed"),
                                  col(fc_lstm, "wind_speed"))
ax.set_title("London — held-out 7-day wind-speed forecast vs observation",
             fontsize=11)
ax.set_ylabel("Wind speed (km/h)")
style_dates(ax)
ax.legend(loc="upper right", framealpha=0.95)
ax.text(
    0.99, 0.04,
    f"7-day MAE — RF: {mae_rf_w:.2f} km/h   |   LSTM: {mae_lstm_w:.2f} km/h",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"),
)
plt.tight_layout()
plt.savefig("figures/fig6b_wind_backtest.png", bbox_inches="tight", dpi=140)
plt.close()
print("Saved figures/fig6b_wind_backtest.png")

# --- Figure 8: RF recursive-collapse fix (temp + wind subpanels) ---
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
for ax, var, ylabel, title in [
    (axL, "temp", "Temperature (°C)", "Temperature"),
    (axR, "wind_speed", "Wind speed (km/h)", "Wind speed"),
]:
    ax.plot(dates, col(df_actual, var), "o-", color="#2196F3",
            linewidth=2.2, label="Observed (Meteostat)")
    ax.plot(dates, col(fc_rf_pre, var), "x--", color="#C62828",
            linewidth=1.8, label="RF — before fix")
    ax.plot(dates, col(fc_rf_post, var), "s-", color="#2E7D32",
            linewidth=1.8, label="RF — after fix")
    pre_spread = col(fc_rf_pre, var).max() - col(fc_rf_pre, var).min()
    post_spread = col(fc_rf_post, var).max() - col(fc_rf_post, var).min()
    mae_pre = mean_absolute_error(col(df_actual, var), col(fc_rf_pre, var))
    mae_post = mean_absolute_error(col(df_actual, var), col(fc_rf_post, var))
    annot = (
        f"Pre-fix   spread {pre_spread:5.2f}   MAE {mae_pre:5.2f}\n"
        f"Post-fix  spread {post_spread:5.2f}   MAE {mae_post:5.2f}"
    )
    ax.text(
        0.02, 0.98, annot,
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="lightgray"),
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    style_dates(ax)
    ax.legend(loc="lower left" if var == "temp" else "lower right",
              framealpha=0.95)
fig.suptitle(
    f"Random Forest recursive 7-day forecast — collapse-fix back-test "
    f"(London, {dates[0].date()} - {dates[-1].date()})",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig("figures/fig8_rf_collapse_fix_backtest.png",
            bbox_inches="tight", dpi=140)
plt.close()
print("Saved figures/fig8_rf_collapse_fix_backtest.png")

print()
print(
    f"MAE summary: RF temp={mae_rf:.2f}°C wind={mae_rf_w:.2f}km/h  "
    f"LSTM temp={mae_lstm:.2f}°C wind={mae_lstm_w:.2f}km/h"
)
print(
    f"RF temp spread: pre="
    f"{col(fc_rf_pre,'temp').max()-col(fc_rf_pre,'temp').min():.2f}  "
    f"post={col(fc_rf_post,'temp').max()-col(fc_rf_post,'temp').min():.2f}"
)
print(
    f"RF wind spread: pre="
    f"{col(fc_rf_pre,'wind_speed').max()-col(fc_rf_pre,'wind_speed').min():.2f}  "
    f"post={col(fc_rf_post,'wind_speed').max()-col(fc_rf_post,'wind_speed').min():.2f}"
)
