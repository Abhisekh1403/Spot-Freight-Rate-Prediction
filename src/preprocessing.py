from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on Earth in miles."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3956.0 * c

def engineer_hybrid_features(df, median_weight=None):
    df = df.copy()

    # 1. Weight Imputation & Flags
    if median_weight is None:
        median_weight = df['weight'].median() if 'weight' in df.columns else 32000.0
    
    df['weight_missing'] = df['weight'].isna().astype('int8')
    df['weight'] = df['weight'].fillna(median_weight).abs()
    
    # 2. Distance Ratios & Spatial Features
    df['log_distance'] = np.log1p(df['distance'])
    df['weight_per_mile'] = df['weight'] / df['distance'].replace(0, 1)
    df['abs_lat_diff'] = (df['pickup_lat'] - df['delivery_lat']).abs()
    df['abs_lon_diff'] = (df['pickup_lon'] - df['delivery_lon']).abs()
    df['pickup_to_delivery_dist'] = haversine_np(
        df['pickup_lon'], df['pickup_lat'], df['delivery_lon'], df['delivery_lat']
    )

    # 3. Distance to Major Freight Hubs
    hubs = {
        'Chicago': (41.8781, -87.6298),
        'Dallas': (32.7767, -96.7970),
        'Atlanta': (33.7490, -84.3880),
        'LA': (34.0522, -118.2437)
    }
    for hub_name, (hub_lat, hub_lon) in hubs.items():
        df[f'dist_pickup_to_{hub_name}'] = haversine_np(df['pickup_lon'], df['pickup_lat'], hub_lon, hub_lat)
        df[f'dist_delivery_to_{hub_name}'] = haversine_np(df['delivery_lon'], df['delivery_lat'], hub_lon, hub_lat)

    # 4. Spatial Grid Zones
    df['pickup_zone'] = df['pickup_lat'].round(1).astype(str) + '_' + df['pickup_lon'].round(1).astype(str)
    df['delivery_zone'] = df['delivery_lat'].round(1).astype(str) + '_' + df['delivery_lon'].round(1).astype(str)

    # 5. Temporal Features & Fourier Harmonics
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['month'] = df['date'].dt.month.astype('float32')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('float32')
    df['day'] = df['date'].dt.day.astype('float32')
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype('int8')

    day_of_year = df['date'].dt.dayofyear.astype('float32')
    for h in range(1, 3):
        angle = 2.0 * np.pi * h * day_of_year / 365.25
        df[f'doy_sin_{h}'] = np.sin(angle)
        df[f'doy_cos_{h}'] = np.cos(angle)
    
    dow_angle = 2.0 * np.pi * df['dayofweek'] / 7.0
    df['dow_sin'] = np.sin(dow_angle)
    df['dow_cos'] = np.cos(dow_angle)

    # 6. Categorical Formatting
    for col in ['pickup', 'delivery', 'equipment', 'pickup_zone', 'delivery_zone']:
        df[col] = df[col].astype(str)

    return df, median_weight

def process_data():
    print("Loading datasets...")
    train_path = DATA_DIR / "train-test.csv"
    val_path = DATA_DIR / "validation.csv"
    dec_path = DATA_DIR / "december-chart-inputs.csv"

    train = pd.read_csv(train_path)
    validation = pd.read_csv(val_path)
    december_inputs = pd.read_csv(dec_path)

    # Historical coordinate lookup for December inputs
    historical_coords = pd.concat([train, validation], ignore_index=True)
    pickup_lookup = historical_coords[['pickup', 'pickup_lat', 'pickup_lon']].dropna().drop_duplicates(subset=['pickup'])
    delivery_lookup = historical_coords[['delivery', 'delivery_lat', 'delivery_lon']].dropna().drop_duplicates(subset=['delivery'])

    december_inputs = december_inputs.drop(columns=['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon'], errors='ignore')
    december_inputs = december_inputs.merge(pickup_lookup, on='pickup', how='left')
    december_inputs = december_inputs.merge(delivery_lookup, on='delivery', how='left')

    # Fallback coordinate defaults
    december_inputs['pickup_lat'] = december_inputs['pickup_lat'].fillna(train['pickup_lat'].median())
    december_inputs['pickup_lon'] = december_inputs['pickup_lon'].fillna(train['pickup_lon'].median())
    december_inputs['delivery_lat'] = december_inputs['delivery_lat'].fillna(train['delivery_lat'].median())
    december_inputs['delivery_lon'] = december_inputs['delivery_lon'].fillna(train['delivery_lon'].median())

    print("Engineering hybrid spatial & temporal features...")
    train_processed, median_weight = engineer_hybrid_features(train)
    val_processed, _ = engineer_hybrid_features(validation, median_weight=median_weight)
    dec_processed, _ = engineer_hybrid_features(december_inputs, median_weight=median_weight)

    return train_processed, val_processed, dec_processed