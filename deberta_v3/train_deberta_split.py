"""
Train DeBERTa-v3-base for nutrient estimation using the pre-split train dataset.
Uses train TXT/JSON files, splits into 90% train / 10% val internally.
Saves scaler parameters to scaler.json for inference.
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils"))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
from nutrient_utils import parse_txt_ingredients

from deberta_dataset import NutrientDataset
from deberta_training_utils import NutrientMetrics, TARGETS

TXT_FILE = os.path.join(DATA_DIR, 'dataset_training_nutrient_estimation_train.txt')
JSON_FILE = os.path.join(DATA_DIR, 'dataset_training_nutrient_estimation_train.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_deberta_v3_split')
SCALER_FILE = os.path.join(BASE_DIR, 'scaler.json')

MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = len(TARGETS)
MAX_LEN = 1024
BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-6
SEED = 42


def main():
    print(f"Model: {MODEL_NAME}")
    print(f"Train TXT: {TXT_FILE}")
    print(f"Train JSON: {JSON_FILE}")

    # 1. Load data
    print("Loading data...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    raw_ingredients = parse_txt_ingredients(TXT_FILE)

    if len(raw_ingredients) != len(json_data):
        print(f"Warning: TXT ({len(raw_ingredients)}) != JSON ({len(json_data)})")
        min_len = min(len(raw_ingredients), len(json_data))
        raw_ingredients = raw_ingredients[:min_len]
        json_data = json_data[:min_len]

    # 2. Prepare texts and labels
    texts = [", ".join(ings) for ings in raw_ingredients]
    labels = []
    for entry in json_data:
        nutrients = entry.get('nutrients_per_100g', {})
        labels.append([nutrients.get(t, 0.0) for t in TARGETS])

    labels_np = np.array(labels, dtype=np.float32)
    print(f"Total training samples: {len(texts)}")

    # 3. Compute scaler from training data
    min_vals = labels_np.min(axis=0)
    max_vals = labels_np.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    print(f"Scaler min: {min_vals}")
    print(f"Scaler range: {range_vals}")

    scaler_data = {
        'min': min_vals.tolist(),
        'range': range_vals.tolist(),
        'targets': TARGETS,
    }
    with open(SCALER_FILE, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    print(f"Scaler saved to {SCALER_FILE}")

    labels_scaled = (labels_np - min_vals) / range_vals

    # 4. Train/Val split (90/10)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels_scaled, test_size=0.1, random_state=SEED
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # 5. Tokenizer & Model
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="regression",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} / {total_params:,} trainable")

    # 6. Datasets
    train_dataset = NutrientDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = NutrientDataset(X_val, y_val, tokenizer, MAX_LEN)

    # 7. Trainer
    compute_metrics = NutrientMetrics(scaler={'min': min_vals, 'range': range_vals})

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        bf16=True,
        max_grad_norm=0.5,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 8. Train
    print("Starting training...")
    trainer.train()

    # 9. Save best model
    best_dir = os.path.join(OUTPUT_DIR, 'best_model')
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best model saved to {best_dir}")

    # 10. Final eval on val
    print("\nFinal validation results:")
    val_results = trainer.evaluate()
    for k, v in sorted(val_results.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
