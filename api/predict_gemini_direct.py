import os
import sys
import json
import random
import time
from tqdm import tqdm
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils"))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

from llm_utils import SYSTEM_PROMPT as SYSTEM, parse_nutrient_string, parse_inst_lines

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running this script.")

GEMINI_MODEL = "gemini-2.5-flash"

TRAIN_TXT = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.txt")
TRAIN_JSON = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.json")
TEST_TXT  = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.txt")
TEST_JSON = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.json")


def get_nutrients_from_gemini(client, recipe_text, examples=None, max_retries=5):
    if "[INST]" in recipe_text:
        user_prompt = recipe_text
    else:
        user_prompt = f"[INST] Determine the nutrient composition per 100 grams in a recipe containing these ingredients: {recipe_text} [/INST]"

    # 建立 multi-turn contents（few-shot + 實際問題）
    contents = []
    if examples:
        for ex_input, ex_output in examples:
            if "[INST]" not in ex_input:
                ex_input = f"[INST] Determine the nutrient composition per 100 grams in a recipe containing these ingredients: {ex_input} [/INST]"
            contents.append(types.Content(role="user",  parts=[types.Part(text=ex_input)]))
            contents.append(types.Content(role="model", parts=[types.Part(text=ex_output)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))

    backoff = 2
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )
            return response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                print(f"  Rate limit, waiting {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                print(f"  Gemini error: {e}")
                return None
    return None


def main():
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 載入 few-shot 訓練資料
    print("Loading train data for few-shot...")
    train_lines = parse_inst_lines(TRAIN_TXT)
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    if len(train_lines) != len(train_data):
        min_len = min(len(train_lines), len(train_data))
        train_lines = train_lines[:min_len]
        train_data  = train_data[:min_len]

    print("Loading test data...")
    test_lines = parse_inst_lines(TEST_TXT)
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    assert len(test_lines) == len(test_data), \
        f"txt ({len(test_lines)}) vs json ({len(test_data)}) 筆數不符"

    print(f"Test samples: {len(test_data)}")

    out_path = os.path.join(BASE_DIR, "prediction_gemini_direct_fewshot.json")

    # 支援斷點續跑
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["case_id"] for r in results}
        print(f"Resuming: {len(results)} already done.")
    else:
        results = []
        done_ids = set()

    for i, (line, item) in enumerate(tqdm(zip(test_lines, test_data), total=len(test_data))):
        case_id = item["case_id"]
        if case_id in done_ids:
            continue

        # 隨機抽 2 筆 few-shot 範例
        indices = random.sample(range(len(train_lines)), 2)
        few_shot = []
        for idx in indices:
            nutrients = train_data[idx].get("nutrients_per_100g", {})
            ex_output = (
                f"Nutrient values per 100 g: "
                f"fat - {nutrients.get('fat_g', 0)}, "
                f"protein - {nutrients.get('protein_g', 0)}, "
                f"saturates - {nutrients.get('saturates_g', 0)}, "
                f"sugars - {nutrients.get('sugars_g', 0)}"
            )
            few_shot.append((train_lines[idx], ex_output))

        answer = get_nutrients_from_gemini(client, line, examples=few_shot)

        if answer:
            parsed = parse_nutrient_string(answer)
            results.append({
                "case_id": case_id,
                "answer": answer,
                "nutrients_per_100g": parsed
            })
        else:
            results.append({
                "case_id": case_id,
                "answer": "",
                "nutrients_per_100g": {}
            })

        # 每 20 筆存一次
        if (i + 1) % 20 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
