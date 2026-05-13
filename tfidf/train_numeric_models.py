"""
Numeric (DictVectorizer-based) nutrient regression with XGBoost & RandomForest.

Features: per-recipe {ingredient_name_unit -> total_quantity} dict.
Pass --use-tfidf to additionally weight the count matrix with TfidfTransformer.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils"))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

from nutrient_utils import evaluate_accuracy
from tfidf_utils import build_numeric_features, vectorize_numeric

TXT_FILE = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_deduplicated.txt")
JSON_FILE = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_deduplicated.json")

TARGETS = ["energy_kcal", "fat_g", "protein_g", "salt_g", "saturates_g", "sugars_g"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-tfidf", action="store_true",
                        help="Apply TfidfTransformer on top of DictVectorizer counts.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Using TfidfTransformer: {args.use_tfidf}")

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    print("Extracting numeric features...")
    feature_dicts, json_data = build_numeric_features(TXT_FILE, json_data)

    y_data = {t: [] for t in TARGETS}
    for entry in json_data:
        nutrients = entry.get("nutrients_per_100g", {})
        for t in TARGETS:
            y_data[t].append(nutrients.get(t, 0.0))
    df_y = pd.DataFrame(y_data)

    # Split: 70% Train / 10% Val / 20% Test
    X_tv, X_test_raw, y_tv, y_test = train_test_split(
        feature_dicts, df_y, test_size=0.2, random_state=args.seed
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.125, random_state=args.seed
    )
    print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    X_train, X_val, X_test, _, _ = vectorize_numeric(
        X_train_raw, X_val_raw, X_test_raw, use_tfidf=args.use_tfidf
    )
    print(f"Feature matrix shape (Train): {X_train.shape}")

    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=args.seed, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=args.seed, n_jobs=-1),
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        df_pred = pd.DataFrame(y_pred, columns=TARGETS, index=y_test.index)

        print(f"Results for {name} on Test Set:")
        for target in TARGETS:
            rmse = np.sqrt(mean_squared_error(y_test[target], df_pred[target]))
            if target in ["protein_g", "sugars_g", "fat_g", "saturates_g"]:
                acc = evaluate_accuracy(y_test[target].values, df_pred[target].values, target)
                acc_str = f"{acc:.2%}"
            else:
                acc_str = "N/A"
            print(f"  {target}: RMSE = {rmse:.4f}, Accuracy = {acc_str}")


if __name__ == "__main__":
    main()
