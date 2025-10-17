# === Load & Clean Data ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from catboost import CatBoostRegressor
import warnings
from tqdm import tqdm
import time
import joblib
from datetime import datetime
import json
import os
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/kaggle/input/ashw-tdi-salah/Gabungan_Surabaya_Sidoarjo.csv')
print(f"Original dataset size: {len(df)}")

# Clean SPP column
df['SPP'] = pd.to_numeric(df['SPP'], errors='coerce').fillna(0).astype(int)
df = df[df['SPP'] > 0]
print(f"After removing SPP=0: {len(df)}")

# OPTION: Uncomment these lines to apply additional filters
#df = df[df['Peserta Didik'] >= 100]
#print(f"After student filter (>=100): {len(df)}")
# df = df[df['SPP'] <= 3300000]
# print(f"After SPP cap (<=3.3M): {len(df)}")

# Change Kurikulum for specific Yayasan
#df.loc[df['Yayasan'] == 'YAYASAN PEMBINA UNIVERSITAS NEGERI JAKARTA', 'Kurikulum'] = 'Kurikulum Internasional'

# Drop columns
columns_to_drop = ['Yayasan', 'Website', 'NPSN']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Fill missing values with mean
columns_to_fill = ['meter listrik', 'harga_listrik_per_bulan', 'estimasi harga tanah sekolah', 
                   'Tanggal SK Pendirian', 'Umur', 'luas tanah']
for col in columns_to_fill:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    mean_value = df[df[col] != 0][col].mean()
    df[col] = df[col].replace(0, mean_value)
    df[col] = df[col].fillna(mean_value)

# Feature engineering
df['rombel_luas_ratio'] = df['Rombel'] / df['luas tanah']
df['total_rooms'] = df['R.Kelas'] + df['R.Lab'] + df['R.Perpus']
df['total_rooms_luas_ratio'] = df['total_rooms'] / df['luas tanah']
df['total_rooms_listrik_ratio'] = df['total_rooms'] / df['meter listrik']
df['listrik_cost_rooms_ratio'] = df['harga_listrik_per_bulan'] / df['total_rooms']

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in df.select_dtypes(include=[np.float64, np.int64]).columns:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Create target variables
df['SPP_log'] = np.log1p(df['SPP'])

print(f"\nFinal dataset size for training: {len(df)}")

# === Data Distribution Analysis ===
print("\n" + "="*50)
print("DATA DISTRIBUTION ANALYSIS")
print("="*50)
print(f"SPP Statistics:")
print(f"  Min: {df['SPP'].min():,.0f}")
print(f"  Max: {df['SPP'].max():,.0f}")
print(f"  Mean: {df['SPP'].mean():,.0f}")
print(f"  Median: {df['SPP'].median():,.0f}")
print(f"\nSPP Distribution:")
print(f"  < 1M: {len(df[df['SPP'] < 1_000_000])} ({len(df[df['SPP'] < 1_000_000])/len(df)*100:.1f}%)")
print(f"  1M-3M: {len(df[(df['SPP'] >= 1_000_000) & (df['SPP'] < 3_000_000)])} ({len(df[(df['SPP'] >= 1_000_000) & (df['SPP'] < 3_000_000)])/len(df)*100:.1f}%)")
print(f"  >= 3M: {len(df[df['SPP'] >= 3_000_000])} ({len(df[df['SPP'] >= 3_000_000])/len(df)*100:.1f}%)")

print(f"\nStudent Count (Peserta Didik):")
print(f"  Min: {df['Peserta Didik'].min():.0f}")
print(f"  Max: {df['Peserta Didik'].max():.0f}")
print(f"  Mean: {df['Peserta Didik'].mean():.0f}")
print(f"  < 100 students: {len(df[df['Peserta Didik'] < 100])} schools")

# === Prepare Features ===
target = 'SPP_log'
exclude_cols = ['SPP', 'SPP_log', 'Nama_sekolah']
features = [col for col in df.columns if col not in exclude_cols]

# Identify categorical features
categorical_features = []
for col in features:
    if col != 'R.Perpus' and (df[col].dtype == 'object' or df[col].nunique() < 10):
        categorical_features.append(col)

print(f"\nCategorical features: {categorical_features}")
print(f"Total features: {len(features)}")

# Prepare X and y (FULL DATASET)
X = df[features].copy()
y = df[target].copy()
y_spp_original = df['SPP'].copy()  # Keep original SPP for range analysis

print(f"\nFull dataset for modeling: {X.shape}")

# Convert categorical features to string
for col in categorical_features:
    X[col] = X[col].astype(str)

# === Helper Functions ===
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)[y_true != 0]) * 100

def calculate_spp_range_metrics(y_true_spp, y_pred_spp):
    """Calculate metrics for different SPP ranges"""
    ranges = {
        'SPP_<1M': (y_true_spp < 1_000_000),
        'SPP_1M-3M': (y_true_spp >= 1_000_000) & (y_true_spp < 3_000_000),
        'SPP_>=3M': (y_true_spp >= 3_000_000)
    }
    
    metrics = {}
    for range_name, mask in ranges.items():
        if mask.sum() > 0:
            y_true_range = y_true_spp[mask]
            y_pred_range = y_pred_spp[mask]
            metrics[f'{range_name}_MAE'] = mean_absolute_error(y_true_range, y_pred_range)
            metrics[f'{range_name}_MAPE'] = mean_absolute_percentage_error(y_true_range, y_pred_range)
            metrics[f'{range_name}_count'] = int(mask.sum())
        else:
            metrics[f'{range_name}_MAE'] = np.nan
            metrics[f'{range_name}_MAPE'] = np.nan
            metrics[f'{range_name}_count'] = 0
    
    return metrics

# === Hyperparameter Grid ===
param_grid = {
    'iterations': [600],
    'learning_rate': [0.05, 0.1, 0.15],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128],
    'random_strength': [0.5, 1.0, 1.5]
}

print(f"\nHyperparameter combinations: {np.prod([len(v) for v in param_grid.values()])}")

# === K-Fold Cross-Validation Setup ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# === Base Model ===
base_model = CatBoostRegressor(
    random_seed=42,
    cat_features=categorical_features,
    verbose=False,
    early_stopping_rounds=50,
    task_type='GPU'
)

# === Custom Scorers ===
def mape_scorer(y_true, y_pred):
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return -mean_absolute_percentage_error(y_true_orig, y_pred_orig)

def mae_original_scorer(y_true, y_pred):
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return -mean_absolute_error(y_true_orig, y_pred_orig)

def r2_original_scorer(y_true, y_pred):
    return r2_score(np.expm1(y_true), np.expm1(y_pred))

scoring = {
    'neg_mae': make_scorer(mae_original_scorer),
    'neg_mape': make_scorer(mape_scorer),
    'r2_log': 'r2',
    'r2_original': make_scorer(r2_original_scorer)
}

# === GridSearchCV on FULL Dataset ===
print("\n" + "="*50)
print("STARTING GRID SEARCH WITH K-FOLD CV ON FULL DATA")
print("="*50)
print(f"This will fit {np.prod([len(v) for v in param_grid.values()]) * cv.n_splits} models")
print(f"Using all {len(X)} samples for cross-validation")
print("Estimated time: 2-4 hours (depending on GPU)")
print("="*50 + "\n")

start_time = time.time()

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=cv,
    scoring=scoring,
    refit='r2_original',
    verbose=2,
    n_jobs=1,
    return_train_score=True
)

grid_search.fit(X, y)

end_time = time.time()
print(f"\n\nGrid Search completed in {(end_time - start_time)/60:.2f} minutes")

# === Extract All Metrics from GridSearch Results ===
print("\n" + "="*50)
print("EXTRACTING METRICS FROM GRID SEARCH RESULTS")
print("="*50)

cv_results = pd.DataFrame(grid_search.cv_results_)

all_results = []
for idx, row in cv_results.iterrows():
    params = row['params']
    
    result = {
        'params': str(params),
        'iterations': params['iterations'],
        'learning_rate': params['learning_rate'],
        'depth': params['depth'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'border_count': params['border_count'],
        'random_strength': params['random_strength'],
        'MAE_mean': -row['mean_test_neg_mae'],
        'MAE_std': row['std_test_neg_mae'],
        'MAPE_mean': -row['mean_test_neg_mape'],
        'MAPE_std': row['std_test_neg_mape'],
        'R2_log_mean': row['mean_test_r2_log'],
        'R2_log_std': row['std_test_r2_log'],
        'R2_original_mean': row['mean_test_r2_original'],
        'R2_original_std': row['std_test_r2_original']
    }
    all_results.append(result)

results_df = pd.DataFrame(all_results)
print(f"✓ Extracted metrics for {len(results_df)} parameter combinations")

# === Best Parameters ===
print("\n" + "="*50)
print("BEST PARAMETERS FROM CV (BY R² ORIGINAL)")
print("="*50)
print(grid_search.best_params_)
print(f"\nBest R² (original): {grid_search.best_score_:.4f}")

# === Save Rankings by Different Metrics ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Sorted by R² original
results_by_r2_orig = results_df.sort_values('R2_original_mean', ascending=False).reset_index(drop=True)
results_by_r2_orig.insert(0, 'Rank', range(1, len(results_by_r2_orig) + 1))
r2_orig_filename = f'all_params_ranked_by_R2_original_{timestamp}.csv'
results_by_r2_orig.to_csv(r2_orig_filename, index=False)
print(f"\n✓ R² (original) rankings saved: {r2_orig_filename}")

# 2. Sorted by R² log
results_by_r2_log = results_df.sort_values('R2_log_mean', ascending=False).reset_index(drop=True)
results_by_r2_log.insert(0, 'Rank', range(1, len(results_by_r2_log) + 1))
r2_log_filename = f'all_params_ranked_by_R2_log_{timestamp}.csv'
results_by_r2_log.to_csv(r2_log_filename, index=False)
print(f"✓ R² (log) rankings saved: {r2_log_filename}")

# 3. Sorted by MAE
results_by_mae = results_df.sort_values('MAE_mean', ascending=True).reset_index(drop=True)
results_by_mae.insert(0, 'Rank', range(1, len(results_by_mae) + 1))
mae_filename = f'all_params_ranked_by_MAE_{timestamp}.csv'
results_by_mae.to_csv(mae_filename, index=False)
print(f"✓ MAE rankings saved: {mae_filename}")

# 4. Sorted by MAPE
results_by_mape = results_df.sort_values('MAPE_mean', ascending=True).reset_index(drop=True)
results_by_mape.insert(0, 'Rank', range(1, len(results_by_mape) + 1))
mape_filename = f'all_params_ranked_by_MAPE_{timestamp}.csv'
results_by_mape.to_csv(mape_filename, index=False)
print(f"✓ MAPE rankings saved: {mape_filename}")

# === Calculate SPP Range Performance for Top 10 Models ===
print("\n" + "="*50)
print("CALCULATING SPP RANGE PERFORMANCE FOR TOP 10 MODELS")
print("="*50)

def calculate_top10_spp_metrics(ranked_df, metric_name):
    """Calculate SPP range metrics for top 10 models"""
    top10_spp_metrics = []
    
    for idx, row in tqdm(ranked_df.head(10).iterrows(), total=10, desc=f"Top 10 by {metric_name}"):
        params = eval(row['params'])
        
        # Create model with these parameters
        model = CatBoostRegressor(
            **params,
            random_seed=42,
            cat_features=categorical_features,
            verbose=False,
            task_type='GPU'
        )
        
        # Get CV predictions
        y_pred_log = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        y_pred_spp = np.expm1(y_pred_log)
        y_true_spp = y_spp_original.values
        
        # Calculate range metrics
        range_metrics = calculate_spp_range_metrics(y_true_spp, y_pred_spp)
        
        # Combine with existing metrics
        combined_metrics = {
            'Rank': row['Rank'],
            'params': row['params'],
            'MAE_mean': row['MAE_mean'],
            'MAE_std': row['MAE_std'],
            'MAPE_mean': row['MAPE_mean'],
            'MAPE_std': row['MAPE_std'],
            'R2_log_mean': row['R2_log_mean'],
            'R2_log_std': row['R2_log_std'],
            'R2_original_mean': row['R2_original_mean'],
            'R2_original_std': row['R2_original_std'],
            **range_metrics
        }
        
        top10_spp_metrics.append(combined_metrics)
    
    return pd.DataFrame(top10_spp_metrics)

# Calculate for each ranking
top10_r2_orig_spp = calculate_top10_spp_metrics(results_by_r2_orig, "R² Original")
top10_r2_log_spp = calculate_top10_spp_metrics(results_by_r2_log, "R² Log")
top10_mae_spp = calculate_top10_spp_metrics(results_by_mae, "MAE")
top10_mape_spp = calculate_top10_spp_metrics(results_by_mape, "MAPE")

# Save top 10 with SPP range metrics
top10_r2_orig_spp.to_csv(f'top10_by_R2_original_with_spp_ranges_{timestamp}.csv', index=False)
top10_r2_log_spp.to_csv(f'top10_by_R2_log_with_spp_ranges_{timestamp}.csv', index=False)
top10_mae_spp.to_csv(f'top10_by_MAE_with_spp_ranges_{timestamp}.csv', index=False)
top10_mape_spp.to_csv(f'top10_by_MAPE_with_spp_ranges_{timestamp}.csv', index=False)

print(f"\n✓ Top 10 models with SPP range analysis saved")

# === Train and Save Top 10 Models for Each Metric ===
print("\n" + "="*50)
print("TRAINING AND SAVING TOP 10 MODELS FOR EACH METRIC")
print("="*50)

os.makedirs(f'top_models_by_R2_original_{timestamp}', exist_ok=True)
os.makedirs(f'top_models_by_R2_log_{timestamp}', exist_ok=True)
os.makedirs(f'top_models_by_MAE_{timestamp}', exist_ok=True)
os.makedirs(f'top_models_by_MAPE_{timestamp}', exist_ok=True)

# Function to train and save models
def train_and_save_top10(ranked_df_with_spp, folder_name, metric_suffix):
    models_info = []
    
    for idx, row in tqdm(ranked_df_with_spp.iterrows(), total=len(ranked_df_with_spp), desc=f"Training {metric_suffix}"):
        params = eval(row['params'])
        
        model = CatBoostRegressor(
            **params,
            random_seed=42,
            cat_features=categorical_features,
            verbose=False,
            task_type='GPU'
        )
        model.fit(X, y)
        
        model_info = row.to_dict()
        
        # Create filename
        if 'R2' in metric_suffix:
            metric_val = row['R2_original_mean'] if 'original' in metric_suffix else row['R2_log_mean']
            metric_std = row['R2_original_std'] if 'original' in metric_suffix else row['R2_log_std']
            filename = f"rank_{int(row['Rank'])}_{metric_suffix}_{metric_val:.4f}_std_{metric_std:.4f}.pkl"
        else:
            metric_val = row['MAE_mean'] if metric_suffix == 'MAE' else row['MAPE_mean']
            metric_std = row['MAE_std'] if metric_suffix == 'MAE' else row['MAPE_std']
            filename = f"rank_{int(row['Rank'])}_{metric_suffix}_{metric_val:.2f}_std_{metric_std:.2f}.pkl"
        
        model_path = os.path.join(folder_name, filename)
        joblib.dump({'model': model, 'info': model_info}, model_path)
        models_info.append(model_info)
    
    # Save summary
    with open(os.path.join(folder_name, 'summary.json'), 'w') as f:
        json.dump(models_info, f, indent=4)
    
    return models_info

# Train all top 10 models
print("\nTraining Top 10 by R² (original)...")
train_and_save_top10(top10_r2_orig_spp, f'top_models_by_R2_original_{timestamp}', 'R2_original')

print("\nTraining Top 10 by R² (log)...")
train_and_save_top10(top10_r2_log_spp, f'top_models_by_R2_log_{timestamp}', 'R2_log')

print("\nTraining Top 10 by MAE...")
train_and_save_top10(top10_mae_spp, f'top_models_by_MAE_{timestamp}', 'MAE')

print("\nTraining Top 10 by MAPE...")
train_and_save_top10(top10_mape_spp, f'top_models_by_MAPE_{timestamp}', 'MAPE')

# === Display Top 10 with SPP Range Metrics ===
print("\n" + "="*80)
print("TOP 10 MODELS BY R² (ORIGINAL) - WITH SPP RANGE PERFORMANCE")
print("="*80)
display_cols = ['Rank', 'R2_original_mean', 'R2_original_std', 'MAE_mean', 
                'SPP_<1M_MAE', 'SPP_1M-3M_MAE', 'SPP_>=3M_MAE']
print(top10_r2_orig_spp[display_cols].to_string(index=False))

print("\n" + "="*80)
print("TOP 10 MODELS BY R² (LOG) - WITH SPP RANGE PERFORMANCE")
print("="*80)
display_cols = ['Rank', 'R2_log_mean', 'R2_log_std', 'MAE_mean',
                'SPP_<1M_MAE', 'SPP_1M-3M_MAE', 'SPP_>=3M_MAE']
print(top10_r2_log_spp[display_cols].to_string(index=False))

print("\n" + "="*80)
print("TOP 10 MODELS BY MAE - WITH SPP RANGE PERFORMANCE")
print("="*80)
display_cols = ['Rank', 'MAE_mean', 'MAE_std', 'R2_original_mean',
                'SPP_<1M_MAE', 'SPP_1M-3M_MAE', 'SPP_>=3M_MAE']
print(top10_mae_spp[display_cols].to_string(index=False))

print("\n" + "="*80)
print("TOP 10 MODELS BY MAPE - WITH SPP RANGE PERFORMANCE")
print("="*80)
display_cols = ['Rank', 'MAPE_mean', 'MAPE_std', 'R2_original_mean',
                'SPP_<1M_MAPE', 'SPP_1M-3M_MAPE', 'SPP_>=3M_MAPE']
print(top10_mape_spp[display_cols].to_string(index=False))

# === Train Final Best Model ===
best_params = eval(top10_r2_orig_spp.iloc[0]['params'])
print("\n" + "="*50)
print("TRAINING FINAL BEST MODEL (BY R² ORIGINAL)")
print("="*50)

final_model = CatBoostRegressor(
    **best_params,
    random_seed=42,
    cat_features=categorical_features,
    verbose=100,
    task_type='GPU'
)

final_model.fit(X, y)

# Get predictions for visualization
y_pred_log_best = cross_val_predict(final_model, X, y, cv=cv, n_jobs=1)
y_actual_spp = np.expm1(y.values)
y_predicted_spp = np.expm1(y_pred_log_best)

# === Feature Importance ===
feature_importance = final_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("TOP 10 FEATURE IMPORTANCE")
print("="*50)
print(importance_df.head(10).to_string(index=False))

# === Visualization ===
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Predictions
predictions_df = pd.DataFrame({
    'Real_SPP': y_actual_spp,
    'Predicted_SPP': y_predicted_spp
})

axes[0].scatter(predictions_df['Real_SPP'], predictions_df['Predicted_SPP'], alpha=0.6)
axes[0].set_xlabel('Real SPP')
axes[0].set_ylabel('Predicted SPP')
axes[0].set_title('Real vs Predicted SPP - Best Model (Log-Log Scale)')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

min_spp = predictions_df[['Real_SPP', 'Predicted_SPP']].min().min()
max_spp = predictions_df[['Real_SPP', 'Predicted_SPP']].max().max()
axes[0].plot([min_spp, max_spp], [min_spp, max_spp], 'r--', label='Perfect Prediction', linewidth=2)
axes[0].legend()

# Plot 2: Feature Importance
top_features = importance_df.head(15)
axes[1].barh(range(len(top_features)), top_features['importance'])
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features['feature'])
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Top 15 Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f'model_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# === Save Additional Files ===
model_filename = f'best_model_by_R2_{timestamp}.pkl'
joblib.dump(final_model, model_filename)
print(f"\n✓ Best model saved: {model_filename}")

feature_filename = f'model_features_{timestamp}.pkl'
joblib.dump({
    'features': features,
    'categorical_features': categorical_features
}, feature_filename)
print(f"✓ Features saved: {feature_filename}")

importance_filename = f'feature_importance_{timestamp}.csv'
importance_df.to_csv(importance_filename, index=False)
print(f"✓ Feature importance saved: {importance_filename}")

# === Final Summary ===
print("\n" + "="*80)
print("TRAINING COMPLETE - COMPREHENSIVE SUMMARY")
print("="*80)
print(f"\nTotal parameter combinations tested: {len(results_df)}")
print(f"Total models trained and saved: 40 (10 per metric)")

print(f"\n📊 FILES CREATED:")
print(f"\n1. RANKING FILES (All {len(results_df)} combinations):")
print(f"   ✓ {r2_orig_filename}")
print(f"   ✓ {r2_log_filename}")
print(f"   ✓ {mae_filename}")
print(f"   ✓ {mape_filename}")

print(f"\n2. TOP 10 WITH SPP RANGE ANALYSIS:")
print(f"   ✓ top10_by_R2_original_with_spp_ranges_{timestamp}.csv")
print(f"   ✓ top10_by_R2_log_with_spp_ranges_{timestamp}.csv")
print(f"   ✓ top10_by_MAE_with_spp_ranges_{timestamp}.csv")
print(f"   ✓ top10_by_MAPE_with_spp_ranges_{timestamp}.csv")

print(f"\n3. MODEL FOLDERS (40 trained models):")
print(f"   ✓ top_models_by_R2_original_{timestamp}/ (10 models + summary.json)")
print(f"   ✓ top_models_by_R2_log_{timestamp}/ (10 models + summary.json)")
print(f"   ✓ top_models_by_MAE_{timestamp}/ (10 models + summary.json)")
print(f"   ✓ top_models_by_MAPE_{timestamp}/ (10 models + summary.json)")

print(f"\n4. BEST MODEL & SUPPORTING FILES:")
print(f"   ✓ {model_filename}")
print(f"   ✓ {feature_filename}")
print(f"   ✓ {importance_filename}")
print(f"   ✓ model_performance_{timestamp}.png")

print("\n" + "="*80)
print("SPP RANGE METRICS INCLUDED FOR TOP 10:")
print("="*80)
print("For each top 10 model, the following SPP range metrics are calculated:")
print("  • SPP < 1M: MAE, MAPE, count")
print("  • SPP 1M-3M: MAE, MAPE, count")
print("  • SPP >= 3M: MAE, MAPE, count")

print("\n" + "="*80)
print("HOW TO USE THE RESULTS")
print("="*80)
print("""
1. ANALYZE SPP RANGE PERFORMANCE:
   - Open top10_by_[METRIC]_with_spp_ranges_[timestamp].csv files
   - Compare how models perform across different SPP ranges
   - Choose models that perform well in your target SPP range

2. LOAD A TRAINED MODEL:
   import joblib
   import numpy as np
   
   # Load a specific model
   model_data = joblib.load('top_models_by_R2_original_[timestamp]/rank_1_[...].pkl')
   model = model_data['model']
   info = model_data['info']
   
   # Check SPP range performance
   print(f"SPP <1M MAE: {info['SPP_<1M_MAE']:.2f}")
   print(f"SPP 1M-3M MAE: {info['SPP_1M-3M_MAE']:.2f}")
   print(f"SPP >=3M MAE: {info['SPP_>=3M_MAE']:.2f}")
   
   # Make predictions
   predictions_log = model.predict(X_new)
   predictions_spp = np.expm1(predictions_log)

3. CHOOSE THE BEST MODEL FOR YOUR USE CASE:
   - If you care about all SPP ranges equally → Use top model by R² or MAE
   - If you care about low SPP schools → Check SPP_<1M_MAE in the CSV
   - If you care about high SPP schools → Check SPP_>=3M_MAE in the CSV
   - If you want stability → Choose models with low std values

4. VIEW DETAILED METRICS:
   - All ranking CSVs show mean and std for all metrics
   - Top 10 CSVs include SPP range breakdown
   - summary.json files contain complete model information
""")

print("\n" + "="*80)
print("EXAMPLE: FINDING BEST MODEL FOR LOW SPP SCHOOLS (<1M)")
print("="*80)
print("""
import pandas as pd

# Load top 10 by MAE with SPP ranges
df = pd.read_csv('top10_by_MAE_with_spp_ranges_[timestamp].csv')

# Sort by SPP_<1M_MAE to find best for low SPP schools
df_sorted = df.sort_values('SPP_<1M_MAE')
print(df_sorted[['Rank', 'SPP_<1M_MAE', 'SPP_<1M_MAPE', 'MAE_mean']].head())

# Load that specific model
best_for_low_spp = df_sorted.iloc[0]
# Use the rank to find the model file in the appropriate folder
""")

print("\n" + "="*80)
print("KEY INSIGHTS FROM SPP RANGE ANALYSIS")
print("="*80)

# Calculate and display insights
for metric_name, df_spp in [
    ("R² Original", top10_r2_orig_spp),
    ("R² Log", top10_r2_log_spp),
    ("MAE", top10_mae_spp),
    ("MAPE", top10_mape_spp)
]:
    print(f"\nTop model by {metric_name}:")
    top_model = df_spp.iloc[0]
    print(f"  Overall MAE: {top_model['MAE_mean']:.2f}")
    print(f"  SPP <1M MAE: {top_model['SPP_<1M_MAE']:.2f} ({top_model['SPP_<1M_count']} schools)")
    print(f"  SPP 1M-3M MAE: {top_model['SPP_1M-3M_MAE']:.2f} ({top_model['SPP_1M-3M_count']} schools)")
    print(f"  SPP >=3M MAE: {top_model['SPP_>=3M_MAE']:.2f} ({top_model['SPP_>=3M_count']} schools)")

print("\n" + "="*80)
print("TRAINING COMPLETE ✓")
print("="*80)
print(f"\nAll files saved with timestamp: {timestamp}")
print("Check the output files to select the best model for your specific use case!")
print("\n" + "="*80)
