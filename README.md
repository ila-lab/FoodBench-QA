# FoodBench-QA

Predict per-100g nutrient values (`fat_g`, `protein_g`, `saturates_g`, `sugars_g`; some methods also output `energy_kcal` and `salt_g`) from a recipe's ingredient list.

This project benchmarks several approaches on the same dataset: **TF-IDF + Ridge**, **TF-IDF + XGBoost/RandomForest**, **DeBERTa-v3** fine-tuning, and **LLMs** (Gemini / local Ollama) with few-shot prompting.

---

## Data

Located in [data/](data/). Every sample is a paired `.txt` / `.json` record, line-aligned:

- `.txt`: one record per line, `[INST] ... ingredient list ... [/INST] answer string`
- `.json`: `[{case_id, answer, nutrients_per_100g{...}}, ...]`

| File                                                             | Rows   | Purpose                       |
| ---------------------------------------------------------------- | ------ | ----------------------------- |
| `dataset_training_nutrient_estimation_deduplicated.{txt,json}` | 14,512 | Full deduplicated set         |
| `dataset_training_nutrient_estimation_train.{txt,json}`        | 11,609 | 80% training set              |
| `dataset_training_nutrient_estimation_test.{txt,json}`         | 2,903  | 20% held-out test set         |
| `T1.1 - dataset_ingredinets_RE.txt`                            | —     | Original raw (pre-dedup) data |

The train/test split is produced from the deduplicated set with `random_state=42` by [data/split_dataset.py](data/split_dataset.py):

```bash
python data/split_dataset.py
```

---

## Project layout

```
FoodBench-QA/
├── data/         dataset + split script
├── EDA/          exploratory analysis & cleaning
├── utils/        shared modules (nutrient utils, TF-IDF, LLM helpers)
├── tfidf/        TF-IDF based methods
├── deberta_v3/   DeBERTa-v3 fine-tuning
├── api/          cloud LLMs (Gemini)
└── llm_ollama/   local Ollama
```

`utils/` is a library, not an entry point. It provides:

- `nutrient_utils.py`: `parse_txt_ingredients`, `normalize_ingredient`, `extract_quantity`, `extract_unit`, `evaluate_accuracy`
- `tfidf_utils.py`: text TF-IDF (word + char n-gram FeatureUnion), Ridge regression, DictVectorizer quantity features, category prototype helpers
- `llm_utils.py`: `SYSTEM_PROMPT`, `parse_nutrient_string`, `parse_inst_lines`

---

## Running the methods

> Every method writes its predictions JSON under the same `nutrients_per_100g` schema, so they can all be scored by the same evaluation script (see [Evaluation](#evaluation) below).

### 1. TF-IDF + Ridge (text features)

Word + character n-gram TF-IDF FeatureUnion, followed by log1p-transformed Ridge regression. Optional blending with per-category prototypes:

```bash
python tfidf/predict_tfidf.py --mode pure        # plain TF-IDF + Ridge, 4 nutrients
python tfidf/predict_tfidf.py --mode prototype   # 0.7 * Ridge + 0.3 * category mean, 6 nutrients
```

Output: `tfidf/prediction_tfidf_<mode>.json`

### 2. TF-IDF + XGBoost / RandomForest (quantity features)

Each recipe is converted into a `{ingredient_name_unit: total_quantity}` dict, vectorized with `DictVectorizer` (optionally re-weighted by `TfidfTransformer`), then fed to XGBoost and RandomForest:

```bash
python tfidf/train_numeric_models.py             # DictVectorizer counts only
python tfidf/train_numeric_models.py --use-tfidf # add TfidfTransformer on top
```

RMSE and accuracy are printed to stdout (no JSON output).

### 3. DeBERTa-v3 fine-tuning

Fine-tunes `microsoft/deberta-v3-base` with a regression head. Internal 90/10 split is taken from the train set; the fitted scaler is saved for inference:

```bash
python deberta_v3/train_deberta_split.py     # train, saves to deberta_v3/results_deberta_v3_split/best_model/
python deberta_v3/predict_and_evaluate.py    # predict on test set + print RMSE / accuracy
```

Output: `deberta_v3/dataset_test_predictions_deberta_v3.json`

### 4. Gemini API (few-shot, direct or TF-IDF-hinted)

Set `GEMINI_API_KEY` in your environment before running. Each query randomly samples 2 few-shot examples from the train set:

```bash
export GEMINI_API_KEY="..."                 # PowerShell: $env:GEMINI_API_KEY = "..."
python api/predict_gemini_direct.py         # plain few-shot
python api/predict_gemini_tfidf.py          # run TF-IDF first, inject the estimate into the prompt (cached in tfidf_preds_cache.json)
```

Output: `api/prediction_gemini_direct_fewshot.json` or `api/prediction_gemini_tfidf_fewshot.json`. Both scripts support resume-on-restart.

### 5. Local Ollama LLM

Start `ollama serve` and `ollama pull` the model first. Model name comes from `--model` (or `$OLLAMA_MODEL`, default `gpt-oss:20b`):

```bash
python llm_ollama/predict_nutrients.py                       # default model
python llm_ollama/predict_nutrients.py --model gemma3:27b    # override
```

Output: `llm_ollama/dataset_test_predictions_ollama.json`

---

## Evaluation

Every method emits a JSON of the form `[{case_id, ..., nutrients_per_100g{...}}, ...]`, so a single script scores any of them. Pass the prediction file and the ground-truth file as CLI arguments:

```bash
python utils/calculate_test_metrics.py \
    --pred deberta_v3/dataset_test_predictions_deberta_v3.json \
    --gt   data/dataset_training_nutrient_estimation_test.json
```

If `nutrients_per_100g` is missing from a prediction record, the script falls back to parsing the `answer` field with `llm_utils.parse_nutrient_string`.

`evaluate_accuracy()` uses per-nutrient tolerance bands (see `check_*_tolerance` in [utils/nutrient_utils.py](utils/nutrient_utils.py)):

| Nutrient                    | Tolerance                                  |
| --------------------------- | ------------------------------------------ |
| `protein_g`, `sugars_g` | ≤10 → ±2; 10–40 → ±20%; >40 → ±8   |
| `fat_g`                   | ≤10 → ±1.5; 10–40 → ±20%; >40 → ±8 |
| `saturates_g`             | <4 → ±0.8; ≥4 → ±20%                  |

---

## EDA

```bash
python EDA/check_duplicates.py     # detect duplicate / inconsistent recipes, write duplicate_recipes_report.csv and the *_deduplicated.{txt,json} pair
python EDA/count_ingredients.py    # normalize and count ingredients, write unique_ingredients_list.csv and normalization_debug.csv
python EDA/analyze_data.py         # ingredient / nutrient distributions, top-20 ingredients, correlation heatmap, outliers → EDA/analysis_results/
```
