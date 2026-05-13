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

from tfidf_utils import parse_ingredient_questions, fit_text_tfidf, predict_text_tfidf
from llm_utils import SYSTEM_PROMPT as SYSTEM, parse_nutrient_string, parse_inst_lines

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running this script.")

GEMINI_MODEL = "gemini-2.5-flash"

TRAIN_TXT  = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.txt")
TRAIN_JSON = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_train.json")
TEST_TXT   = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.txt")
TEST_JSON  = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_test.json")

NUTRIENT_KEYS = ["protein_g", "fat_g", "sugars_g", "saturates_g"]

# ========== Gemini 相關 ==========

def build_prompt_with_tfidf(recipe_text, tfidf_pred):
    """把原始 [INST] prompt + TF-IDF 預測值組合成新 prompt"""
    hint = (
        f"[TF-IDF Reference Estimate] "
        f"fat - {tfidf_pred['fat_g']:.2f}, "
        f"protein - {tfidf_pred['protein_g']:.2f}, "
        f"saturates - {tfidf_pred['saturates_g']:.2f}, "
        f"sugars - {tfidf_pred['sugars_g']:.2f}"
    )

    # 在 [/INST] 前插入 TF-IDF 提示
    if "[/INST]" in recipe_text:
        prompt = recipe_text.replace("[/INST]", f"\n{hint}\n[/INST]")
    else:
        prompt = f"{recipe_text}\n{hint}"

    return prompt


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
    TFIDF_CACHE = os.path.join(BASE_DIR, "tfidf_preds_cache.json")

    test_lines = parse_inst_lines(TEST_TXT)
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    assert len(test_lines) == len(test_data), \
        f"txt ({len(test_lines)}) vs json ({len(test_data)}) 筆數不符"

    # ---------- Step 1+2: TF-IDF（有 cache 就直接 load） ----------
    if os.path.exists(TFIDF_CACHE):
        print("=== Loading TF-IDF predictions from cache ===")
        with open(TFIDF_CACHE, "r", encoding="utf-8") as f:
            tfidf_preds = json.load(f)
        print(f"Cache loaded: {len(tfidf_preds)} items.")
        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            train_data = json.load(f)
    else:
        print("=== Step 1: Training TF-IDF models ===")
        train_questions = parse_ingredient_questions(TRAIN_TXT)
        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        featurizer, tfidf_models = fit_text_tfidf(train_questions, train_data, NUTRIENT_KEYS)

        print("\n=== Step 2: TF-IDF predictions on test set ===")
        test_questions = parse_ingredient_questions(TEST_TXT)
        tfidf_preds = predict_text_tfidf(featurizer, tfidf_models, test_questions, NUTRIENT_KEYS)

        with open(TFIDF_CACHE, "w", encoding="utf-8") as f:
            json.dump(tfidf_preds, f)
        print(f"TF-IDF cache saved to {TFIDF_CACHE}")

    print(f"TF-IDF predictions ready for {len(tfidf_preds)} items.")

    # 載入訓練集 [INST] lines，供 few-shot 使用
    train_inst_lines = parse_inst_lines(TRAIN_TXT)
    if len(train_inst_lines) != len(train_data):
        min_len = min(len(train_inst_lines), len(train_data))
        train_inst_lines = train_inst_lines[:min_len]

    # ---------- Step 3: 送 Gemini ----------
    print("\n=== Step 3: Gemini predictions with TF-IDF hints ===")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    out_path = os.path.join(BASE_DIR, "prediction_gemini_tfidf_fewshot.json")

    # 支援斷點續跑（空筆也會重試）
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # 只把有成功答案的筆視為已完成
        done_ids = {r["case_id"] for r in results if r.get("answer")}
        # 移除空筆，讓它們被重新填入
        results = [r for r in results if r.get("answer")]
        print(f"Resuming: {len(results)} done, will retry empty entries.")
    else:
        results = []
        done_ids = set()

    for i, (line, item, tfidf_pred) in enumerate(
        tqdm(zip(test_lines, test_data, tfidf_preds), total=len(test_data))
    ):
        case_id = item["case_id"]
        if case_id in done_ids:
            continue

        # 隨機抽 2 筆 few-shot 範例（不加 TF-IDF hint，保持範例簡潔）
        indices = random.sample(range(len(train_inst_lines)), 2)
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
            few_shot.append((train_inst_lines[idx], ex_output))

        augmented_prompt = build_prompt_with_tfidf(line, tfidf_pred)
        answer = get_nutrients_from_gemini(gemini_client, augmented_prompt, examples=few_shot)

        if answer:
            parsed = parse_nutrient_string(answer)
            results.append({
                "case_id": case_id,
                "answer": answer,
                "nutrients_per_100g": parsed,
                "tfidf_hint": tfidf_pred
            })
        else:
            results.append({
                "case_id": case_id,
                "answer": "",
                "nutrients_per_100g": {},
                "tfidf_hint": tfidf_pred
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
