"""
Text-based TF-IDF + Ridge nutrient prediction.

Modes:
  pure       : 4 nutrients, plain TF-IDF + Ridge
  prototype  : 6 nutrients, 0.7*ML + 0.3*category-prototype
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils"))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

from tfidf_utils import (
    parse_ingredient_questions,
    fit_text_tfidf,
    predict_text_tfidf,
    build_prototypes,
    food_category,
)

TRAIN_TXT = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.txt")
TRAIN_JSON = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.json")
TEST_TXT = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.txt")
TEST_JSON = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.json")

NUTRIENTS_PURE = ["protein_g", "fat_g", "sugars_g", "saturates_g"]
NUTRIENTS_PROTOTYPE = ["energy_kcal", "protein_g", "fat_g", "sugars_g", "saturates_g", "salt_g"]


def main():
    parser = argparse.ArgumentParser(description="Text-based TF-IDF nutrient prediction.")
    parser.add_argument(
        "--mode",
        choices=["pure", "prototype"],
        default="pure",
        help="pure = plain TF-IDF + Ridge; prototype = blend with category prototype",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    nutrient_keys = NUTRIENTS_PURE if args.mode == "pure" else NUTRIENTS_PROTOTYPE
    out_path = args.output or os.path.join(BASE_DIR, f"prediction_tfidf_{args.mode}.json")

    train_questions = parse_ingredient_questions(TRAIN_TXT)
    test_questions = parse_ingredient_questions(TEST_TXT)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    assert len(train_questions) == len(train_data), \
        f"Train .txt ({len(train_questions)}) vs .json ({len(train_data)}) mismatch"
    assert len(test_questions) == len(test_data), \
        f"Test .txt ({len(test_questions)}) vs .json ({len(test_data)}) mismatch"

    print(f"Mode: {args.mode}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test  samples: {len(test_data)}")

    featurizer, models = fit_text_tfidf(train_questions, train_data, nutrient_keys)

    print("\nPredicting on TEST...")
    ml_preds = predict_text_tfidf(featurizer, models, test_questions, nutrient_keys)

    if args.mode == "prototype":
        prototypes = build_prototypes(train_questions, train_data, nutrient_keys)
        global_mean_proto = prototypes.get("general")

    predictions = []
    for i, item in enumerate(tqdm(test_data)):
        pred = ml_preds[i]
        if args.mode == "prototype":
            cat = food_category(test_questions[i])
            proto = prototypes.get(cat, global_mean_proto)
            pred = {k: max(0.0, 0.7 * pred[k] + 0.3 * proto[k]) for k in nutrient_keys}
        predictions.append({
            "case_id": item["case_id"],
            "nutrients_per_100g": pred,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved: {out_path}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
