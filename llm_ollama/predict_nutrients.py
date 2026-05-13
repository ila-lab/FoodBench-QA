"""
Predict nutrient values on the test set using a local Ollama model
with few-shot prompting from the training set.
"""

import argparse
import json
import os
import random
import sys

import ollama
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils"))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

from llm_utils import SYSTEM_PROMPT as SYSTEM, parse_nutrient_string, parse_inst_lines


def get_nutrients_from_ollama(recipe_text, examples, model):
    messages = [{"role": "system", "content": SYSTEM}]

    for ex_input, ex_output in examples:
        if "[INST]" not in ex_input:
            ex_input = (
                f"[INST] Determine the nutrient composition per 100 grams "
                f"in a recipe containing these ingredients: {ex_input} [/INST]"
            )
        messages.append({"role": "user", "content": ex_input})
        messages.append({"role": "assistant", "content": ex_output})

    if "[INST]" not in recipe_text:
        recipe_text = (
            f"[INST] Determine the nutrient composition per 100 grams "
            f"in a recipe containing these ingredients: {recipe_text} [/INST]"
        )
    messages.append({"role": "user", "content": recipe_text})

    try:
        response = ollama.chat(model=model, messages=messages, options={"temperature": 0})
        return response["message"]["content"]
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def format_example_output(nutrients):
    return (
        f"Nutrient values per 100 g: "
        f"fat - {nutrients.get('fat_g', 0)}, "
        f"protein - {nutrients.get('protein_g', 0)}, "
        f"saturates - {nutrients.get('saturates_g', 0)}, "
        f"sugars - {nutrients.get('sugars_g', 0)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
                        help="Ollama model tag (default from $OLLAMA_MODEL or 'gpt-oss:20b')")
    parser.add_argument("--num-few-shot", type=int, default=2)
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "dataset_test_predictions_ollama.json"))
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N predictions.")
    args = parser.parse_args()

    train_txt = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.txt")
    train_json = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.json")
    test_txt = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.txt")
    test_json = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.json")

    print(f"Model: {args.model}")
    print("Loading data...")
    train_lines = parse_inst_lines(train_txt)
    test_lines = parse_inst_lines(test_txt)
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if len(train_lines) != len(train_data):
        min_len = min(len(train_lines), len(train_data))
        train_lines, train_data = train_lines[:min_len], train_data[:min_len]
    if len(test_lines) != len(test_data):
        min_len = min(len(test_lines), len(test_data))
        test_lines, test_data = test_lines[:min_len], test_data[:min_len]

    print(f"Train pool: {len(train_lines)}, test items: {len(test_lines)}")

    results = []
    for i, line in tqdm(enumerate(test_lines), total=len(test_lines)):
        case_id = test_data[i].get("case_id", str(i + 1))

        indices = random.sample(range(len(train_lines)), args.num_few_shot)
        examples = [
            (train_lines[idx], format_example_output(train_data[idx].get("nutrients_per_100g", {})))
            for idx in indices
        ]

        answer = get_nutrients_from_ollama(line, examples, model=args.model)
        if answer:
            results.append({
                "case_id": case_id,
                "answer": answer,
                "nutrients_per_100g": parse_nutrient_string(answer),
            })

        if args.save_every and (i + 1) % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
