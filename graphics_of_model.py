# ================== Learning Curve + dll ==================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib, warnings, os, json, time
from sklearn.model_selection import KFold, learning_curve, cross_val_predict
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# --------- CONFIG: update these 3 paths to your saved files ----------
DATA_PATH = "/kaggle/input/ashw-tdi-salah/Gabungan_Surabaya_Sidoarjo.csv"
BEST_MODEL_PATH = "/kaggle/input/fixed-10-fold-no-filter/top_models_by_MAE_20251008_134550/rank_1_MAE_688901.55_std_291912.05.pkl"   # <- ganti filenya sesuai hasil cv
FEATURES_META_PATH = "/kaggle/input/fixed-10-fold-no-filter/model_features_20251008_134550.pkl"  # <- ganti filenya sesuai hasil cv
# ---------------------------------------------------------------------

# === Load saved model
_loaded = joblib.load(BEST_MODEL_PATH)
model_saved = _loaded["model"] if isinstance(_loaded, dict) and "model" in _loaded else _loaded

# === Load feature metadata (features + categorical_features)
meta = joblib.load(FEATURES_META_PATH)
features = meta["features"]
categorical_features = meta["categorical_features"]

df = pd.read_csv(DATA_PATH)

# Clean SPP
df["SPP"] = pd.to_numeric(df["SPP"], errors="coerce").fillna(0).astype(int)
df = df[df["SPP"] > 0]
#df = df[df["Peserta Didik"] >= 100]

# Whitelist
df.loc[df["Yayasan"] == "YAYASAN PEMBINA UNIVERSITAS NEGERI JAKARTA", "Kurikulum"] = "Kurikulum Internasional"

# Drop cols
df = df.drop(columns=["Yayasan", "Website", "NPSN"], errors="ignore")

# Fill missing numeric cols with mean (excluding zeros)
to_fill = ["meter listrik","harga_listrik_per_bulan","estimasi harga tanah sekolah",
           "Tanggal SK Pendirian","Umur","luas tanah"]
for col in to_fill:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    mean_value = df[df[col] != 0][col].mean()
    df[col] = df[col].replace(0, mean_value).fillna(mean_value)

# Feature engineering (same)
df["rombel_luas_ratio"] = df["Rombel"] / df["luas tanah"]
df["total_rooms"] = df["R.Kelas"] + df["R.Lab"] + df["R.Perpus"]
df["total_rooms_luas_ratio"] = df["total_rooms"] / df["luas tanah"]
df["total_rooms_listrik_ratio"] = df["total_rooms"] / df["meter listrik"]
df["listrik_cost_rooms_ratio"] = df["harga_listrik_per_bulan"] / df["total_rooms"]

# Replace inf/NaN with medians
df = df.replace([np.inf, -np.inf], np.nan)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Targets (same)
df["Buying Power"] = df["SPP"] * df["Peserta Didik"]
df["Buying Power_log"] = np.log1p(df["Buying Power"])
df["SPP_log"] = np.log1p(df["SPP"])

target = "SPP_log"
exclude = ["SPP","SPP_log","Buying Power","Buying Power_log","Nama_sekolah"]
X = df[[c for c in df.columns if c not in exclude]].copy()
y = df[target].copy()

# Ensure categorical columns are str (CatBoost by name)
for c in categorical_features:
    if c in X.columns:
        X[c] = X[c].astype(str)

print(f"Prepared data: X={X.shape}, y={y.shape}")

# === Custom scorers (original-scale)
def r2_original_scorer(y_true, y_pred):
    return r2_score(np.expm1(y_true), np.expm1(y_pred))

def r2_log_scorer(y_true, y_pred):
    return r2_score(y_true, y_pred)

def mae_original_scorer(y_true, y_pred):
    y_true_orig, y_pred_orig = np.expm1(y_true), np.expm1(y_pred)
    return -mean_absolute_error(y_true_orig, y_pred_orig)  # neg for "higher-is-better"

def mape_original_scorer(y_true, y_pred):
    y_true_orig, y_pred_orig = np.expm1(y_true), np.expm1(y_pred)
    return -mean_absolute_percentage_error(y_true_orig, y_pred_orig)  # neg for "higher-is-better"

# === K-Fold (same seed/shuffle as training)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# === Build an unfitted clone with same hyperparams as the saved model
best_params = model_saved.get_params(deep=False)
best_params.update(dict(cat_features=categorical_features, verbose=False, task_type=best_params.get("task_type","GPU")))
lc_model = CatBoostRegressor(**best_params)

# In CV, the max effective train fraction per fold is (K-1)/K
max_frac = (cv.n_splits - 1) / cv.n_splits
train_sizes = np.linspace(0.1, max_frac, 12)

# ========== 1. GET PREDICTIONS FOR SCATTER PLOTS ==========
print("\nGenerating cross-validated predictions...")
y_pred_log = cross_val_predict(lc_model, X, y, cv=cv, n_jobs=1)
y_pred_original = np.expm1(y_pred_log)
y_true_original = np.expm1(y)

# Calculate metrics
r2_orig = r2_score(y_true_original, y_pred_original)
mae_orig = mean_absolute_error(y_true_original, y_pred_original)
mape_orig = mean_absolute_percentage_error(y_true_original, y_pred_original)

print(f"CV Metrics - R²: {r2_orig:.4f}, MAE: {mae_orig:.2f}, MAPE: {mape_orig:.4f}")

# ========== 2. PREDICTED VS ACTUAL (REAL SCALE) ==========
plt.figure(figsize=(8,8))
plt.scatter(y_true_original, y_pred_original, alpha=0.5, s=20)
min_val = min(y_true_original.min(), y_pred_original.min())
max_val = max(y_true_original.max(), y_pred_original.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual SPP (Original Scale)')
plt.ylabel('Predicted SPP (Original Scale)')
plt.title(f'Predicted vs Actual (Original Scale)\nR² = {r2_orig:.4f}, MAE = {mae_orig:.2f}')
plt.legend()
plt.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = f"pred_vs_actual_original_{ts}.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Predicted vs Actual (original) saved to: {out_png}")

# ========== 3. PREDICTED VS ACTUAL (LOG SCALE) ==========
plt.figure(figsize=(8,8))
plt.scatter(y, y_pred_log, alpha=0.5, s=20)
min_val = min(y.min(), y_pred_log.min())
max_val = max(y.max(), y_pred_log.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual SPP (Log Scale)')
plt.ylabel('Predicted SPP (Log Scale)')
r2_log = r2_score(y, y_pred_log)
plt.title(f'Predicted vs Actual (Log Scale)\nR² = {r2_log:.4f}')
plt.legend()
plt.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = f"pred_vs_actual_log_{ts}.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Predicted vs Actual (log) saved to: {out_png}")

# ========== 4. RESIDUAL PLOT (ORIGINAL SCALE) ==========
residuals = y_true_original - y_pred_original

plt.figure(figsize=(10,6))
plt.scatter(y_pred_original, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted SPP (Original Scale)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot (Original Scale)')
plt.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = f"residual_plot_original_{ts}.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Residual plot (original) saved to: {out_png}")

# ========== 5. RESIDUAL PLOT (LOG SCALE) ==========
residuals_log = y - y_pred_log

plt.figure(figsize=(10,6))
plt.scatter(y_pred_log, residuals_log, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted SPP (Log Scale)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot (Log Scale)')
plt.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = f"residual_plot_log_{ts}.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Residual plot (log) saved to: {out_png}")

# ========== ORIGINAL LEARNING CURVES ==========

# === Learning curve (R² original scale)
print("\nGenerating learning curve (R² original scale)...")
ts_abs, tr_scores, va_scores = learning_curve(
    estimator=lc_model,
    X=X, y=y,
    cv=cv,
    scoring=make_scorer(r2_original_scorer),
    train_sizes=train_sizes,
    n_jobs=1,
    shuffle=False
)

tr_mean, tr_std = tr_scores.mean(axis=1), tr_scores.std(axis=1)
va_mean, va_std = va_scores.mean(axis=1), va_scores.std(axis=1)

# Plot
plt.figure(figsize=(8,5))
plt.title("Learning Curve (Saved CatBoost) — R² (original scale)")
plt.plot(ts_abs, tr_mean, linestyle="--", label="Training score")
plt.plot(ts_abs, va_mean, label="Cross-validation score")
plt.fill_between(ts_abs, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15)
plt.fill_between(ts_abs, va_mean-va_std, va_mean+va_std, alpha=0.15)
plt.xlabel("Training set size (samples)")
plt.ylabel("R² (original)")
plt.legend()
plt.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = f"learning_curve_r2_original_{ts}.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Learning curve (R² original) saved to: {out_png}")

# === Learning curve (R² log scale) ===
print("\nGenerating learning curve (R² log scale)...")
ts_abs_log, tr_scores_log, va_scores_log = learning_curve(
    estimator=lc_model,
    X=X, y=y,
    cv=cv,
    scoring=make_scorer(r2_log_scorer),
    train_sizes=train_sizes,
    n_jobs=1,
    shuffle=False
)

tr_mean_log = tr_scores_log.mean(axis=1)
tr_std_log = tr_scores_log.std(axis=1)
va_mean_log = va_scores_log.mean(axis=1)
va_std_log = va_scores_log.std(axis=1)

# Plot
plt.figure(figsize=(8,5))
plt.title("Learning Curve (Saved CatBoost) — R² (log scale)")
plt.plot(ts_abs_log, tr_mean_log, linestyle="--", label="Training score")
plt.plot(ts_abs_log, va_mean_log, label="Cross-validation score")
plt.fill_between(ts_abs_log, tr_mean_log-tr_std_log, tr_mean_log+tr_std_log, alpha=0.15)
plt.fill_between(ts_abs_log, va_mean_log-va_std_log, va_mean_log+va_std_log, alpha=0.15)
plt.xlabel("Training set size (samples)")
plt.ylabel("R² (log)")
plt.legend()
plt.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = f"learning_curve_r2_log_{ts}.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Learning curve (R² log) saved to: {out_png}")

# === Learning curve (MAE on original scale) ===
print("\nGenerating learning curve (MAE)...")
ts_abs_mae, tr_scores_mae, va_scores_mae = learning_curve(
    estimator=lc_model,
    X=X, y=y,
    cv=cv,
    scoring=make_scorer(mae_original_scorer),
    train_sizes=train_sizes,
    n_jobs=1,
    shuffle=False
)

# Convert to POSITIVE MAE
tr_mean_mae = (-tr_scores_mae).mean(axis=1)
tr_std_mae  = tr_scores_mae.std(axis=1)
va_mean_mae = (-va_scores_mae).mean(axis=1)
va_std_mae  = va_scores_mae.std(axis=1)

# Plot
plt.figure(figsize=(8,5))
plt.title("Learning Curve (Saved CatBoost) — MAE (original scale)")
plt.plot(ts_abs_mae, tr_mean_mae, linestyle="--", label="Training MAE")
plt.plot(ts_abs_mae, va_mean_mae, label="CV MAE")
plt.fill_between(ts_abs_mae, tr_mean_mae-tr_std_mae, tr_mean_mae+tr_std_mae, alpha=0.15)
plt.fill_between(ts_abs_mae, va_mean_mae-va_std_mae, va_mean_mae+va_std_mae, alpha=0.15)
plt.xlabel("Training set size (samples)")
plt.ylabel("MAE (lower is better)")
plt.legend()
plt.tight_layout()
ts_mae = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png_mae = f"learning_curve_mae_original_{ts_mae}.png"
plt.savefig(out_png_mae, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Learning curve (MAE) saved to: {out_png_mae}")

# ========== 5. LEARNING CURVE (MAPE - ORIGINAL SCALE) ==========
print("\nGenerating learning curve (MAPE)...")
ts_abs_mape, tr_scores_mape, va_scores_mape = learning_curve(
    estimator=lc_model,
    X=X, y=y,
    cv=cv,
    scoring=make_scorer(mape_original_scorer),
    train_sizes=train_sizes,
    n_jobs=1,
    shuffle=False
)

# Convert to POSITIVE MAPE (as percentage)
tr_mean_mape = (-tr_scores_mape).mean(axis=1) * 100
tr_std_mape  = tr_scores_mape.std(axis=1) * 100
va_mean_mape = (-va_scores_mape).mean(axis=1) * 100
va_std_mape  = va_scores_mape.std(axis=1) * 100

# Plot
plt.figure(figsize=(8,5))
plt.title("Learning Curve (Saved CatBoost) — MAPE (original scale)")
plt.plot(ts_abs_mape, tr_mean_mape, linestyle="--", label="Training MAPE")
plt.plot(ts_abs_mape, va_mean_mape, label="CV MAPE")
plt.fill_between(ts_abs_mape, tr_mean_mape-tr_std_mape, tr_mean_mape+tr_std_mape, alpha=0.15)
plt.fill_between(ts_abs_mape, va_mean_mape-va_std_mape, va_mean_mape+va_std_mape, alpha=0.15)
plt.xlabel("Training set size (samples)")
plt.ylabel("MAPE % (lower is better)")
plt.legend()
plt.tight_layout()
ts_mape = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png_mape = f"learning_curve_mape_original_{ts_mape}.png"
plt.savefig(out_png_mape, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Learning curve (MAPE) saved to: {out_png_mape}")

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print(f"Total plots created: 9")
print("="*60)
