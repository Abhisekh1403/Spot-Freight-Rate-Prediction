from pathlib import Path
import joblib
import numpy as np
from catboost import CatBoostRegressor
from preprocessing import process_data

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_model():
    train_df, _, _ = process_data()

    feature_cols = [
        'distance', 'log_distance', 'weight', 'weight_missing', 'weight_per_mile',
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'abs_lat_diff', 'abs_lon_diff', 'pickup_to_delivery_dist',
        'dist_pickup_to_Chicago', 'dist_delivery_to_Chicago',
        'dist_pickup_to_Dallas', 'dist_delivery_to_Dallas',
        'dist_pickup_to_Atlanta', 'dist_delivery_to_Atlanta',
        'dist_pickup_to_LA', 'dist_delivery_to_LA',
        'month', 'dayofweek', 'day', 'is_weekend',
        'doy_sin_1', 'doy_cos_1', 'doy_sin_2', 'doy_cos_2',
        'dow_sin', 'dow_cos',
        'equipment', 'pickup', 'delivery', 'pickup_zone', 'delivery_zone'
    ]

    cat_features = ['equipment', 'pickup', 'delivery', 'pickup_zone', 'delivery_zone']

    # Rate-Per-Mile (RPM) target transformation in log space
    train_df['rate_per_mile'] = train_df['posted_rate'] / train_df['distance']
    train_df['target_log_rpm'] = np.log1p(train_df['rate_per_mile'])

    best_params = {
        'loss_function': 'RMSE',
        'iterations': 1200,
        'learning_rate': 0.04,
        'depth': 8,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'task_type': 'CPU',
        'verbose': 100
    }

    print("Training CatBoost on Rate-Per-Mile target with hybrid feature set...")
    model = CatBoostRegressor(**best_params)
    model.fit(
        train_df[feature_cols],
        train_df['target_log_rpm'],
        cat_features=cat_features
    )

    joblib.dump(model, MODELS_DIR / "freight_rate_model.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    print(f"\nModel successfully saved to {MODELS_DIR}")

if __name__ == "__main__":
    train_model()