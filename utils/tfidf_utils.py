"""
Shared TF-IDF utilities for nutrient prediction.

Two feature pipelines are provided:

1. text-based  (`build_text_featurizer` + `train_ridge_models`)
   Word + character n-gram TfidfVectorizer FeatureUnion, used in
   `tfidf/predict_tfidf.py` and `api/predict_gemini_tfidf.py`.

2. numeric / DictVectorizer based (`build_numeric_features`)
   Quantity-aware features keyed by "<name>_<unit>", optionally weighted
   by TfidfTransformer. Used in `tfidf/train_numeric_models.py`.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor

from nutrient_utils import normalize_ingredient, parse_txt_ingredients, extract_quantity, extract_unit


# ---------------------------------------------------------------------------
# Shared text parsing
# ---------------------------------------------------------------------------

def parse_ingredient_questions(txt_path):
    """Extract the [INST]...[/INST] body from each line of a dataset txt file."""
    questions = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r"\[INST\](.*?)\[/INST\]", line)
            questions.append(m.group(1).strip() if m else line)
    return questions


# ---------------------------------------------------------------------------
# Text-based pipeline (word + char n-gram TF-IDF + Ridge)
# ---------------------------------------------------------------------------

def build_text_featurizer():
    """Return the word + char n-gram FeatureUnion used across all text models."""
    word_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=8000,
    )
    char_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=12000,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def train_ridge_models(X_train, train_data, nutrient_keys, alphas=None, verbose=True):
    """Train one RidgeCV (log1p-transformed target) per nutrient key.

    train_data is a list of dicts containing nutrients_per_100g.
    Returns dict[nutrient] -> fitted TransformedTargetRegressor.
    """
    if alphas is None:
        alphas = np.logspace(-2, 3, 15)

    targets = {k: [] for k in nutrient_keys}
    for item in train_data:
        for k in nutrient_keys:
            targets[k].append(item["nutrients_per_100g"][k])

    models = {}
    for nutrient in nutrient_keys:
        y = np.array(targets[nutrient], dtype=float)
        ridge = RidgeCV(alphas=alphas, cv=5)
        model = TransformedTargetRegressor(
            regressor=ridge,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
        model.fit(X_train, y)
        models[nutrient] = model
        if verbose:
            print(f"  {nutrient:16s} | alpha={model.regressor_.alpha_:.4f}")
    return models


def fit_text_tfidf(train_questions, train_data, nutrient_keys, verbose=True):
    """Convenience wrapper: build featurizer, fit, train ridge models."""
    featurizer = build_text_featurizer()
    if verbose:
        print("Building TF-IDF on ingredient texts...")
    X_train = featurizer.fit_transform(train_questions)
    if verbose:
        print(f"Train feature shape: {X_train.shape}")
    models = train_ridge_models(X_train, train_data, nutrient_keys, verbose=verbose)
    return featurizer, models


def predict_text_tfidf(featurizer, models, questions, nutrient_keys):
    """Run TF-IDF + Ridge predictions, clipped at 0."""
    X = featurizer.transform(questions)
    preds = []
    for i in range(len(questions)):
        pred = {k: max(0.0, float(models[k].predict(X[i])[0])) for k in nutrient_keys}
        preds.append(pred)
    return preds


# ---------------------------------------------------------------------------
# Numeric / DictVectorizer pipeline (quantity-aware, for XGB / RF)
# ---------------------------------------------------------------------------

def build_numeric_features(txt_file, json_data=None):
    """Convert each recipe into a {ingredient_name_unit: total_quantity} dict.

    Returns (feature_dicts, json_data_aligned).
    Caller can later run DictVectorizer (+ optional TfidfTransformer) on
    feature_dicts.
    """
    raw_ingredients_per_recipe = parse_txt_ingredients(txt_file)

    if json_data is not None and len(raw_ingredients_per_recipe) != len(json_data):
        print(
            f"Warning: parsed recipes ({len(raw_ingredients_per_recipe)}) "
            f"!= JSON entries ({len(json_data)})"
        )
        min_len = min(len(raw_ingredients_per_recipe), len(json_data))
        raw_ingredients_per_recipe = raw_ingredients_per_recipe[:min_len]
        json_data = json_data[:min_len]

    feature_dicts = []
    for ingredients in raw_ingredients_per_recipe:
        recipe_features = {}
        for ing_text in ingredients:
            name = normalize_ingredient(ing_text)
            if not name:
                continue
            unit = extract_unit(ing_text)
            qty = extract_quantity(ing_text)
            key = f"{name}_{unit}"
            recipe_features[key] = recipe_features.get(key, 0.0) + qty
        feature_dicts.append(recipe_features)

    return feature_dicts, json_data


def vectorize_numeric(X_train_raw, X_val_raw, X_test_raw, use_tfidf=False):
    """DictVectorize the three splits; optionally apply TfidfTransformer.

    Vectorizer / transformer are fit on X_train_raw only.
    Returns (X_train, X_val, X_test, dict_vectorizer, tfidf_transformer_or_None).
    """
    dict_vec = DictVectorizer(sparse=True)
    X_train = dict_vec.fit_transform(X_train_raw)
    X_val = dict_vec.transform(X_val_raw)
    X_test = dict_vec.transform(X_test_raw)

    tfidf = None
    if use_tfidf:
        tfidf = TfidfTransformer()
        X_train = tfidf.fit_transform(X_train)
        X_val = tfidf.transform(X_val)
        X_test = tfidf.transform(X_test)

    return X_train, X_val, X_test, dict_vec, tfidf


# ---------------------------------------------------------------------------
# Category prototype helpers (used by --mode prototype in predict_tfidf.py)
# ---------------------------------------------------------------------------

PROTEIN_KW = ["chicken", "beef", "fish", "egg", "tofu", "pork", "turkey", "lamb", "tuna", "salmon", "shrimp"]
SWEET_KW = ["cake", "dessert", "cookie", "sugar", "chocolate", "honey", "syrup", "candy", "jam", "brownie"]
FATTY_KW = ["fried", "butter", "oil", "bacon", "cheese", "cream", "lard", "shortening"]


def food_category(text):
    t = text.lower()
    if any(k in t for k in SWEET_KW):
        return "sweet"
    if any(k in t for k in PROTEIN_KW):
        return "protein"
    if any(k in t for k in FATTY_KW):
        return "fatty"
    return "general"


def build_prototypes(questions, data, nutrient_keys):
    """Per-category mean nutrient profile."""
    buckets = {"protein": [], "sweet": [], "fatty": [], "general": []}
    for q, item in zip(questions, data):
        buckets[food_category(q)].append(item["nutrients_per_100g"])
    proto = {}
    for cat, items in buckets.items():
        if items:
            proto[cat] = {k: float(np.mean([d[k] for d in items])) for k in nutrient_keys}
    return proto
