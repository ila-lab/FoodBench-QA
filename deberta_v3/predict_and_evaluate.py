"""
Predict nutrient values on the test set using the trained DeBERTa-v3 model,
then evaluate against ground truth with RMSE and custom accuracy metrics.
"""

import os
import sys
import json
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils"))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
from nutrient_utils import parse_txt_ingredients, evaluate_accuracy

from deberta_dataset import NutrientDataset
from deberta_training_utils import TARGETS

TEST_TXT_FILE = os.path.join(DATA_DIR, 'dataset_training_nutrient_estimation_test.txt')
TEST_JSON_FILE = os.path.join(DATA_DIR, 'dataset_training_nutrient_estimation_test.json')
SCALER_FILE = os.path.join(BASE_DIR, 'scaler.json')
MODEL_DIR = os.path.join(BASE_DIR, 'results_deberta_v3_split', 'best_model')

MAX_LEN = 512
BATCH_SIZE = 16


def main():
    # 1. Load scaler
    print(f"Loading scaler from {SCALER_FILE}...")
    with open(SCALER_FILE, 'r') as f:
        scaler = json.load(f)
    min_vals = np.array(scaler['min'], dtype=np.float32)
    range_vals = np.array(scaler['range'], dtype=np.float32)
    print(f"  min: {min_vals}")
    print(f"  range: {range_vals}")

    # 2. Load model & tokenizer
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32
    )

    # 3. Parse test inputs
    print(f"Parsing test file: {TEST_TXT_FILE}")
    raw_ingredients = parse_txt_ingredients(TEST_TXT_FILE)
    input_texts = [", ".join(ings) for ings in raw_ingredients]
    print(f"  Parsed {len(input_texts)} test samples")

    # 4. Load ground truth
    print(f"Loading ground truth from {TEST_JSON_FILE}...")
    with open(TEST_JSON_FILE, 'r', encoding='utf-8') as f:
        test_json = json.load(f)

    if len(input_texts) != len(test_json):
        print(f"  Warning: TXT ({len(input_texts)}) != JSON ({len(test_json)})")
        min_len = min(len(input_texts), len(test_json))
        input_texts = input_texts[:min_len]
        test_json = test_json[:min_len]

    y_true = []
    for entry in test_json:
        nutrients = entry.get('nutrients_per_100g', {})
        y_true.append([nutrients.get(t, 0.0) for t in TARGETS])
    y_true = np.array(y_true, dtype=np.float32)

    # 5. Run inference (NutrientDataset with labels=None)
    print("Running inference...")
    dataset = NutrientDataset(input_texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, 'temp_inference'),
        per_device_eval_batch_size=BATCH_SIZE,
        bf16=True,
    )
    trainer = Trainer(model=model, args=training_args)

    preds_scaled = trainer.predict(dataset).predictions
    if isinstance(preds_scaled, tuple):
        preds_scaled = preds_scaled[0]

    # 6. Inverse scale
    y_pred = preds_scaled * range_vals + min_vals
    y_pred = np.maximum(y_pred, 0.0)

    # 7. Evaluate
    print("\n" + "=" * 65)
    print("  TEST SET EVALUATION RESULTS")
    print("=" * 65)
    print(f"  {'Nutrient':<15} | {'RMSE':<10} | {'Accuracy':<10}")
    print("-" * 65)

    for i, name in enumerate(TARGETS):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        acc = evaluate_accuracy(true_vals, pred_vals, name)
        print(f"  {name:<15} | {rmse:<10.4f} | {acc:<10.2%}")

    print("=" * 65)
    print(f"  Total test samples: {len(y_true)}")
    print()

    # 8. Save predictions to JSON
    output_file = os.path.join(BASE_DIR, 'dataset_test_predictions_deberta_v3.json')
    output_data = []
    for i in range(len(y_pred)):
        case_id = test_json[i].get('case_id', str(i))
        nutrients = {t: float(round(y_pred[i][j], 2)) for j, t in enumerate(TARGETS)}
        answer_str = (
            f"Nutrient values per 100 g: "
            f"fat - {nutrients['fat_g']}, "
            f"protein - {nutrients['protein_g']}, "
            f"saturates - {nutrients['saturates_g']}, "
            f"sugars - {nutrients['sugars_g']}"
        )
        output_data.append({
            "case_id": case_id,
            "answer": answer_str,
            "nutrients_per_100g": nutrients,
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
