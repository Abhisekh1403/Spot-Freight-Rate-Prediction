from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from preprocessing import process_data

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

def generate_predictions():
    print("Preparing prediction datasets...")
    _, val_df, dec_df = process_data()

    model_path = MODELS_DIR / "freight_rate_model.joblib"
    feature_path = MODELS_DIR / "feature_cols.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    print("Generating validation and December predictions...")
    
    # 1. Predict Log Rate-Per-Mile
    val_pred_log_rpm = model.predict(val_df[feature_cols])
    dec_pred_log_rpm = model.predict(dec_df[feature_cols])

    # 2. Invert RPM back to total dollar rates: rate = expm1(log_rpm) * distance
    val_preds = np.clip(np.expm1(val_pred_log_rpm), 0.01, None) * val_df['distance'].values
    dec_preds = np.clip(np.expm1(dec_pred_log_rpm), 0.01, None) * dec_df['distance'].values

    # 3. Save Validation Predictions
    val_template_path = DATA_DIR / "validation-predictions-template.csv"
    val_output_path = ROOT_DIR / "validation_predictions.csv"
    
    if val_template_path.is_file():
        val_output = pd.read_csv(val_template_path)
        val_output['predicted_rate'] = val_preds
    else:
        val_output = pd.DataFrame({
            "load_id": val_df["load_id"],
            "predicted_rate": val_preds
        })
    
    val_output.to_csv(val_output_path, index=False)
    print(f"Saved {val_output_path}")

    # 4. Save December Chart Inputs
    dec_chart_path = DATA_DIR / "december-chart-inputs.csv"
    dec_df['predicted_rate'] = dec_preds
    dec_df[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']].to_csv(dec_chart_path, index=False)
    print(f"Saved {dec_chart_path}")

if __name__ == "__main__":
    generate_predictions()