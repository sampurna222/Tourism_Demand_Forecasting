# -*- coding: utf-8 -*-
"""
=============================================================================
Tourism Demand Forecasting – Week 4 Predictive Modeling Task
=============================================================================
Approaches implemented:
  1. SARIMA  – classical time-series model
  2. Random Forest Regressor – ensemble ML regression
  3. Gradient Boosting Regressor – boosted tree regression
All results + plots are saved to ./outputs/
=============================================================================
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Setup output directory
# ---------------------------------------------------------------------------
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# ---------------------------------------------------------------------------
# 1. Simulate Realistic Tourism Demand Dataset  (2010-01 -> 2023-12)
# ---------------------------------------------------------------------------
print("=" * 70)
print("  STEP 1 – Data Simulation")
print("=" * 70)

np.random.seed(42)
n_months = 168          # 14 years × 12 months
dates = pd.date_range("2010-01-01", periods=n_months, freq="MS")

# Base trend (logistic growth)
trend = 80_000 + 60_000 * (1 / (1 + np.exp(-0.03 * (np.arange(n_months) - 80))))

# Seasonality  (strong summer peak – tourism-typical)
month = np.array([d.month for d in dates])
seasonality = (
    20_000 * np.sin(2 * np.pi * (month - 3) / 12) +
    8_000  * np.sin(4 * np.pi * (month - 3) / 12)
)

# Exogenous covariates
gdp_growth   = 2.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_months) / 48) + 0.2 * np.random.randn(n_months)
cpi          = 100 + 0.3 * np.arange(n_months) + 0.5 * np.random.randn(n_months)
exchange_rate= 1.2 - 0.001 * np.arange(n_months) + 0.05 * np.random.randn(n_months)
holiday_days = np.where(month.isin([6, 7, 8, 12]) if hasattr(month, 'isin')
                        else np.isin(month, [6, 7, 8, 12]), 1, 0).astype(float)
holiday_days += np.where(np.isin(month, [1, 5]), 0.5, 0)

# COVID shock: 2020-04 -> 2021-06
covid_shock = np.ones(n_months)
for i, d in enumerate(dates):
    if pd.Timestamp("2020-04-01") <= d <= pd.Timestamp("2021-06-01"):
        covid_shock[i] = 0.15 + 0.05 * np.random.rand()
    elif pd.Timestamp("2021-07-01") <= d <= pd.Timestamp("2022-06-01"):
        covid_shock[i] = 0.55 + 0.10 * np.random.rand()

# Noise
noise = 3_000 * np.random.randn(n_months)

arrivals = (trend + seasonality) * covid_shock + noise
arrivals  = np.clip(arrivals, 5_000, None).astype(int)

df = pd.DataFrame({
    "date":          dates,
    "arrivals":      arrivals,
    "month":         month,
    "year":          np.array([d.year for d in dates]),
    "gdp_growth":    gdp_growth,
    "cpi":           cpi,
    "exchange_rate": exchange_rate,
    "holiday_days":  holiday_days,
    "covid_shock":   covid_shock,
})
df.set_index("date", inplace=True)

print(f"Dataset shape : {df.shape}")
print(f"Date range    : {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Arrivals range: {df['arrivals'].min():,} – {df['arrivals'].max():,}")
print(df.describe().round(2))

df.to_csv("outputs/simulated_tourism_dataset.csv")
print("\n  [✓] Dataset saved -> outputs/simulated_tourism_dataset.csv")

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis  &  Time-Series Decomposition
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 2 – Exploratory Data Analysis")
print("=" * 70)

# ── 2a. Time-series overview
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
axes[0].plot(df.index, df["arrivals"], color=PALETTE[0], linewidth=1.5)
axes[0].set_title("Monthly International Tourist Arrivals (2010–2023)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Arrivals")
axes[0].fill_between(df.index, df["arrivals"], alpha=0.15, color=PALETTE[0])

axes[1].plot(df.index, df["gdp_growth"], color=PALETTE[1], linewidth=1.4, label="GDP Growth (%)")
axes[1].plot(df.index, df["exchange_rate"], color=PALETTE[2], linewidth=1.4, label="Exchange Rate")
axes[1].legend(); axes[1].set_ylabel("Economic Indicators")

axes[2].plot(df.index, df["cpi"], color=PALETTE[3], linewidth=1.4, label="CPI")
axes[2].legend(); axes[2].set_ylabel("CPI")
axes[2].set_xlabel("Date")

plt.tight_layout()
plt.savefig("outputs/01_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved -> outputs/01_overview.png")

# ── 2b. Seasonal decomposition
decomp = seasonal_decompose(df["arrivals"], model="additive", period=12)
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
components = [
    (decomp.observed, "Observed", PALETTE[0]),
    (decomp.trend,    "Trend",    PALETTE[1]),
    (decomp.seasonal, "Seasonal", PALETTE[2]),
    (decomp.resid,    "Residual", PALETTE[3]),
]
for ax, (comp, title, color) in zip(axes, components):
    ax.plot(df.index, comp, color=color, linewidth=1.3)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Arrivals")
plt.suptitle("Seasonal Decomposition of Tourist Arrivals", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/02_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved -> outputs/02_decomposition.png")

# ── 2c. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[["arrivals", "gdp_growth", "cpi", "exchange_rate", "holiday_days", "covid_shock"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved -> outputs/03_correlation.png")

# ── 2d. Monthly seasonality box-plot
fig, ax = plt.subplots(figsize=(12, 5))
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df_plot = df.copy(); df_plot["month_name"] = [month_names[m-1] for m in df_plot["month"]]
sns.boxplot(data=df_plot, x="month_name", y="arrivals",
            order=month_names, palette="tab10", ax=ax)
ax.set_title("Tourist Arrivals by Month (Seasonal Pattern)", fontsize=13, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Arrivals")
plt.tight_layout()
plt.savefig("outputs/04_seasonality_box.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved -> outputs/04_seasonality_box.png")

# ── 2e. ADF stationarity test
adf_result = adfuller(df["arrivals"].dropna())
print(f"\n  ADF Test – ADF Stat: {adf_result[0]:.4f} | p-value: {adf_result[1]:.4f}")
adf_diff_result = adfuller(df["arrivals"].diff(12).dropna())
print(f"  ADF Test (12-diff) – ADF Stat: {adf_diff_result[0]:.4f} | p-value: {adf_diff_result[1]:.4f}")

# ACF / PACF
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
plot_acf (df["arrivals"].diff(12).dropna(), lags=36, ax=axes[0, 0], color=PALETTE[0])
axes[0, 0].set_title("ACF – Seasonally Differenced")
plot_pacf(df["arrivals"].diff(12).dropna(), lags=36, ax=axes[0, 1], color=PALETTE[1])
axes[0, 1].set_title("PACF – Seasonally Differenced")
plot_acf (df["arrivals"], lags=36, ax=axes[1, 0], color=PALETTE[2])
axes[1, 0].set_title("ACF – Original")
plot_pacf(df["arrivals"], lags=36, ax=axes[1, 1], color=PALETTE[3])
axes[1, 1].set_title("PACF – Original")
plt.tight_layout()
plt.savefig("outputs/05_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved -> outputs/05_acf_pacf.png")

# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 3 – Feature Engineering")
print("=" * 70)

df["arrivals_lag1"]  = df["arrivals"].shift(1)
df["arrivals_lag12"] = df["arrivals"].shift(12)
df["arrivals_roll3"] = df["arrivals"].shift(1).rolling(3).mean()
df["arrivals_roll6"] = df["arrivals"].shift(1).rolling(6).mean()
df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
df["trend_idx"]      = np.arange(len(df))
df["year_norm"]      = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())
df["gdp_cpi_ratio"]  = df["gdp_growth"] / (df["cpi"] / 100)

features = [
    "arrivals_lag1", "arrivals_lag12", "arrivals_roll3", "arrivals_roll6",
    "month_sin", "month_cos", "trend_idx", "year_norm",
    "gdp_growth", "cpi", "exchange_rate", "holiday_days",
    "covid_shock", "gdp_cpi_ratio",
]
df_ml = df.dropna().copy()
print(f"  Features created : {len(features)}")
print(f"  ML-ready rows    : {len(df_ml)}")
print(f"  Feature list     : {features}")

# ---------------------------------------------------------------------------
# 4. Train / Test Split  (last 24 months = test)
# ---------------------------------------------------------------------------
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"

train = df_ml[df_ml.index <= TRAIN_END]
test  = df_ml[df_ml.index >= TEST_START]

X_train, y_train = train[features], train["arrivals"]
X_test,  y_test  = test[features],  test["arrivals"]

print(f"\n  Train: {train.index[0].date()} -> {train.index[-1].date()}  ({len(train)} obs)")
print(f"  Test : {test.index[0].date()}  -> {test.index[-1].date()}   ({len(test)} obs)")

# ---------------------------------------------------------------------------
# 5.  Model A – SARIMA(1,1,1)(1,1,1)[12]
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 4 – SARIMA Model")
print("=" * 70)

sarima_train = df[df.index <= TRAIN_END]["arrivals"]
sarima_test  = df[df.index >= TEST_START]["arrivals"]

sarima_model = SARIMAX(
    sarima_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_fit = sarima_model.fit(disp=False, maxiter=200)
print(sarima_fit.summary())

sarima_forecast = sarima_fit.get_forecast(steps=len(sarima_test))
sarima_pred     = sarima_forecast.predicted_mean
sarima_ci       = sarima_forecast.conf_int(alpha=0.05)
sarima_pred.index = sarima_test.index

sarima_mae  = mean_absolute_error(sarima_test, sarima_pred)
sarima_rmse = np.sqrt(mean_squared_error(sarima_test, sarima_pred))
sarima_mape = np.mean(np.abs((sarima_test - sarima_pred) / sarima_test)) * 100
sarima_r2   = r2_score(sarima_test, sarima_pred)

print(f"\n  SARIMA Results ->  MAE: {sarima_mae:,.0f}  |  RMSE: {sarima_rmse:,.0f}  |"
      f"  MAPE: {sarima_mape:.2f}%  |  R²: {sarima_r2:.4f}")

# Diagnostic plot
fig = sarima_fit.plot_diagnostics(figsize=(14, 8))
plt.suptitle("SARIMA Model Diagnostics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/06_sarima_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved -> outputs/06_sarima_diagnostics.png")

# ---------------------------------------------------------------------------
# 6.  Model B – Random Forest Regressor
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 5 – Random Forest Regressor")
print("=" * 70)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
rf_r2   = r2_score(y_test, rf_pred)

# Cross-validation (TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=5)
rf_cv_r2 = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="r2")
rf_cv_neg_mae = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

print(f"  RF  Results →  MAE: {rf_mae:,.0f}  |  RMSE: {rf_rmse:,.0f}  |"
      f"  MAPE: {rf_mape:.2f}%  |  R²: {rf_r2:.4f}")
print(f"  CV R² (mean ± std): {rf_cv_r2.mean():.4f} ± {rf_cv_r2.std():.4f}")
print(f"  CV MAE (mean): {-rf_cv_neg_mae.mean():,.0f}")

# Feature importance
feat_imp_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# 7.  Model C – Gradient Boosting Regressor
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 6 – Gradient Boosting Regressor")
print("=" * 70)

gb = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

gb_mae  = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mape = np.mean(np.abs((y_test - gb_pred) / y_test)) * 100
gb_r2   = r2_score(y_test, gb_pred)

gb_cv_r2 = cross_val_score(gb, X_train, y_train, cv=tscv, scoring="r2")
gb_cv_neg_mae = cross_val_score(gb, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

feat_imp_gb = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)

print(f"  GB  Results →  MAE: {gb_mae:,.0f}  |  RMSE: {gb_rmse:,.0f}  |"
      f"  MAPE: {gb_mape:.2f}%  |  R²: {gb_r2:.4f}")
print(f"  CV R² (mean ± std): {gb_cv_r2.mean():.4f} ± {gb_cv_r2.std():.4f}")
print(f"  CV MAE (mean): {-gb_cv_neg_mae.mean():,.0f}")

# ---------------------------------------------------------------------------
# 8. Visualisations
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 7 – Generating Visualisations")
print("=" * 70)

# ── 8a.  Forecast comparison (all three models)
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df.index, df["arrivals"], label="Actual", color="black", linewidth=1.5, alpha=0.8)
ax.plot(sarima_pred.index, sarima_pred, label="SARIMA Forecast", color=PALETTE[0],
        linewidth=1.8, linestyle="--")
ax.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1],
                alpha=0.15, color=PALETTE[0], label="SARIMA 95% CI")
ax.plot(test.index, rf_pred, label="Random Forest", color=PALETTE[1], linewidth=1.8, linestyle="-.")
ax.plot(test.index, gb_pred, label="Gradient Boosting", color=PALETTE[2], linewidth=1.8, linestyle=":")
ax.axvline(pd.Timestamp(TEST_START), color="red", linestyle="--", alpha=0.6, label="Train/Test split")
ax.set_title("Tourism Demand Forecasting – Model Comparison", fontsize=14, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Monthly Arrivals")
ax.legend(loc="upper left"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("outputs/07_forecast_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved → outputs/07_forecast_comparison.png")

# ── 8b.  Actual vs Predicted scatter (test period)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, pred, color) in zip(axes, [
    ("SARIMA",           sarima_pred.values, PALETTE[0]),
    ("Random Forest",    rf_pred,            PALETTE[1]),
    ("Gradient Boosting",gb_pred,            PALETTE[2]),
]):
    actual = sarima_test.values if name == "SARIMA" else y_test.values
    ax.scatter(actual, pred, alpha=0.7, color=color, edgecolors="white", linewidth=0.4, s=60)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
    ax.set_title(f"{name}\nR² = {r2_score(actual, pred):.4f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual Arrivals"); ax.set_ylabel("Predicted Arrivals")
plt.suptitle("Actual vs Predicted – Test Period", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/08_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved → outputs/08_actual_vs_predicted.png")

# ── 8c.  Residuals over time
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
for ax, (name, actual, pred, color) in zip(axes, [
    ("SARIMA",            sarima_test.values, sarima_pred.values, PALETTE[0]),
    ("Random Forest",     y_test.values,      rf_pred,            PALETTE[1]),
    ("Gradient Boosting", y_test.values,      gb_pred,            PALETTE[2]),
]):
    residuals = actual - pred
    ax.plot(residuals, color=color, linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(residuals)), residuals, alpha=0.2, color=color)
    ax.set_title(f"{name} Residuals  (Test Period)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Residual")
plt.tight_layout()
plt.savefig("outputs/09_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved → outputs/09_residuals.png")

# ── 8d.  Feature importance comparison (RF vs GB)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (name, imp, color) in zip(axes, [
    ("Random Forest",     feat_imp_rf, PALETTE[1]),
    ("Gradient Boosting", feat_imp_gb, PALETTE[2]),
]):
    imp.head(10).sort_values().plot(kind="barh", ax=ax, color=color, edgecolor="white")
    ax.set_title(f"Feature Importance – {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/10_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved → outputs/10_feature_importance.png")

# ── 8e.  Cross-validation R² scores (bar chart)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(5)
width = 0.35
ax.bar(x - width/2, rf_cv_r2, width, label="Random Forest", color=PALETTE[1], alpha=0.85, edgecolor="white")
ax.bar(x + width/2, gb_cv_r2, width, label="Gradient Boosting", color=PALETTE[2], alpha=0.85, edgecolor="white")
ax.axhline(rf_cv_r2.mean(), color=PALETTE[1], linestyle="--", linewidth=1.5, alpha=0.7, label=f"RF Mean R²={rf_cv_r2.mean():.3f}")
ax.axhline(gb_cv_r2.mean(), color=PALETTE[2], linestyle="--", linewidth=1.5, alpha=0.7, label=f"GB Mean R²={gb_cv_r2.mean():.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
ax.set_title("Cross-Validation R² Scores (TimeSeriesSplit, k=5)", fontsize=12, fontweight="bold")
ax.set_ylabel("R² Score"); ax.legend()
plt.tight_layout()
plt.savefig("outputs/11_cv_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved → outputs/11_cv_scores.png")

# ── 8f.  Metrics comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
models = ["SARIMA", "Random Forest", "Gradient Boosting"]
maes   = [sarima_mae,  rf_mae,  gb_mae]
rmses  = [sarima_rmse, rf_rmse, gb_rmse]
r2s    = [sarima_r2,   rf_r2,   gb_r2]

for ax, (metric, vals, fmt) in zip(axes, [
    ("MAE",  maes,  lambda v: f"{v:,.0f}"),
    ("RMSE", rmses, lambda v: f"{v:,.0f}"),
    ("R²",   r2s,   lambda v: f"{v:.4f}"),
]):
    bars = ax.bar(models, vals, color=PALETTE[:3], edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.97,
                fmt(val), ha="center", va="top", fontweight="bold", fontsize=9, color="white")
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric); ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=10)
plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/12_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Plot saved → outputs/12_metrics_comparison.png")

# ---------------------------------------------------------------------------
# 9.  Save metrics to JSON for DOC generator
# ---------------------------------------------------------------------------
metrics = {
    "SARIMA": {
        "MAE": round(sarima_mae, 2), "RMSE": round(sarima_rmse, 2),
        "MAPE": round(sarima_mape, 2), "R2": round(sarima_r2, 4),
        "order": "SARIMA(1,1,1)(1,1,1)[12]",
    },
    "Random_Forest": {
        "MAE": round(rf_mae, 2), "RMSE": round(rf_rmse, 2),
        "MAPE": round(rf_mape, 2), "R2": round(rf_r2, 4),
        "CV_R2_mean": round(rf_cv_r2.mean(), 4),
        "CV_R2_std":  round(rf_cv_r2.std(),  4),
        "CV_MAE_mean": round(-rf_cv_neg_mae.mean(), 2),
        "n_estimators": 300, "max_depth": 12,
    },
    "Gradient_Boosting": {
        "MAE": round(gb_mae, 2), "RMSE": round(gb_rmse, 2),
        "MAPE": round(gb_mape, 2), "R2": round(gb_r2, 4),
        "CV_R2_mean": round(gb_cv_r2.mean(), 4),
        "CV_R2_std":  round(gb_cv_r2.std(),  4),
        "CV_MAE_mean": round(-gb_cv_neg_mae.mean(), 2),
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 4,
    },
    "dataset": {
        "n_obs": n_months, "date_range": "2010-01 to 2023-12",
        "train_size": len(train), "test_size": len(test),
        "n_features": len(features),
    },
    "adf_test": {
        "original_stat": round(float(adf_result[0]), 4),
        "original_pvalue": round(float(adf_result[1]), 4),
        "diff12_stat": round(float(adf_diff_result[0]), 4),
        "diff12_pvalue": round(float(adf_diff_result[1]), 4),
    },
    "top_features_rf": feat_imp_rf.head(5).round(4).to_dict(),
    "top_features_gb": feat_imp_gb.head(5).round(4).to_dict(),
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\n  [✓] Metrics saved → outputs/metrics.json")

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  {'Model':<25}  {'MAE':>10}  {'RMSE':>10}  {'MAPE':>8}  {'R²':>8}")
print(f"  {'-'*65}")
for name, mae, rmse, mape, r2 in [
    ("SARIMA(1,1,1)(1,1,1)[12]", sarima_mae, sarima_rmse, sarima_mape, sarima_r2),
    ("Random Forest",            rf_mae,    rf_rmse,    rf_mape,    rf_r2),
    ("Gradient Boosting",        gb_mae,    gb_rmse,    gb_mape,    gb_r2),
]:
    print(f"  {name:<25}  {mae:>10,.0f}  {rmse:>10,.0f}  {mape:>7.2f}%  {r2:>8.4f}")

print("\n  All outputs saved to: ./outputs/")
print("  Run generate_doc.py next to create the Word document report.\n")
