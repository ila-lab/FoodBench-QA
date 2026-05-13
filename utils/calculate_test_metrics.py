"""
Compute RMSE and custom accuracy for any prediction JSON against ground truth.

Both files must be lists of records keyed by `case_id`. The predicted nutrients
are read from `nutrients_per_100g` if present, otherwise parsed from the
`answer` string via llm_utils.parse_nutrient_string.

Usage:
    python utils/calculate_test_metrics.py --pred PATH --gt PATH
"""

import argparse
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error

from nutrient_utils import evaluate_accuracy
from llm_utils import parse_nutrient_string

TARGETS = ['fat_g', 'protein_g', 'saturates_g', 'sugars_g']


def get_predicted_nutrients(entry):
    """Prefer the structured field; fall back to parsing the `answer` string."""
    nut = entry.get('nutrients_per_100g')
    if isinstance(nut, dict) and any(k in nut for k in TARGETS):
        return {t: float(nut.get(t, 0.0)) for t in TARGETS}
    return parse_nutrient_string(entry.get('answer', ''))


def main():
    parser = argparse.ArgumentParser(description="Evaluate nutrient predictions against ground truth.")
    parser.add_argument("--pred", required=True, help="Predictions JSON path")
    parser.add_argument("--gt", required=True, help="Ground truth JSON path")
    args = parser.parse_args()

    print(f"Loading predictions from: {args.pred}")
    if not os.path.exists(args.pred):
        raise FileNotFoundError(args.pred)
    with open(args.pred, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    print(f"Loading ground truth from: {args.gt}")
    if not os.path.exists(args.gt):
        raise FileNotFoundError(args.gt)
    with open(args.gt, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)

    ground_truth_map = {
        item['case_id']: item['nutrients_per_100g']
        for item in ground_truth_data if 'case_id' in item
    }
    print(f"Loaded {len(predictions_data)} predictions and {len(ground_truth_map)} ground truth records.")

    y_true = {t: [] for t in TARGETS}
    y_pred = {t: [] for t in TARGETS}
    missing_truth = 0
    empty_preds = 0

    for entry in predictions_data:
        case_id = entry.get('case_id')
        if case_id not in ground_truth_map:
            missing_truth += 1
            continue

        truth = ground_truth_map[case_id]
        predicted = get_predicted_nutrients(entry)
        if not predicted:
            empty_preds += 1

        for t in TARGETS:
            y_true[t].append(float(truth.get(t, 0.0)))
            y_pred[t].append(float(predicted.get(t, 0.0)))

    print(f"Done. {missing_truth} missing ground truth, {empty_preds} empty predictions.")

    print("\nEvaluation Results:")
    print("-" * 60)
    print(f"{'Nutrient':<15} | {'RMSE':<10} | {'Accuracy':<10}")
    print("-" * 60)
    for t in TARGETS:
        true_vals = np.array(y_true[t])
        pred_vals = np.array(y_pred[t])
        if len(true_vals) == 0:
            print(f"{t:<15} | {'N/A':<10} | {'N/A':<10}")
            continue
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        acc = evaluate_accuracy(true_vals, pred_vals, t)
        print(f"{t:<15} | {rmse:<10.4f} | {acc:<10.2%}")
    print("-" * 60)


if __name__ == "__main__":
    main()
