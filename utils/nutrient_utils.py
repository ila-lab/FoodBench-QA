import re
import os

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------

# Units list
UNITS = [
    'bushel', 'shot', 'glass', 'cup', 'tablespoon', 'tbsp', 'tsp', 'kg', 'teaspoon', 
    'g', 'ml', 'ounce', 'oz', 'pound', 'lb', 'quart', 'pint', 'gallon', 'dash', 'drop', 
    'pinch', 'fl. oz', 'scoop', 'liter', 'l', 'can', 'bottle', 'package', 'stick', 'slice', 
    'clove', 'head', 'stalk', 'sprig', 'bunch', 'piece', 'container', 'jar', 'bag', 'box', 'envelope'
]
# Ensure sorted by length descending to match longest first
UNITS.sort(key=len, reverse=True)

# Target nutrients
TARGETS = ['fat_g', 'protein_g', 'saturates_g', 'sugars_g']

# -------------------------------------------------------------------------
# TEXT PROCESSING & PARSING
# -------------------------------------------------------------------------

def normalize_ingredient(ing):
    """
    Normalizes/cleans an ingredient string.
    Removes quantities, units, and parentheses to isolate the ingredient name.
    """
    # Lowercase and strip
    ing = ing.strip().lower()
    
    # 1. Remove parenthesized content FIRST (e.g. "1 (15 oz) can") -> "1  can"
    ing = re.sub(r'\(.*?\)', '', ing)

    # 1b. Remove "number to number" patterns e.g. "1 to 2"
    ing = re.sub(r'\d+\s*to\s*\d+', '', ing)
    
    # 2. Remove leading specific characters/numbers
    # e.g. "1", "1/2", "1-2", "1.5", "*"
    ing = re.sub(r'^[\d\s/\.\-\*]+', '', ing).strip()
    
    words_to_remove = list(UNITS) # Copy list
    
    # Construct regex: (word1|word2...)(?:es|s)?
    pattern_str = r'^(' + '|'.join(re.escape(w) for w in words_to_remove) + r')(?:es|s)?\b\s*'
    
    prev_ing = None
    while ing != prev_ing:
        prev_ing = ing
        ing = re.sub(pattern_str, '', ing).strip()
    
    # 3. Singularize ingredient
    if ing.endswith('oes'):
        ing = ing[:-2]
    elif ing.endswith('ies'):
        ing = ing[:-3] + 'y'
    elif ing.endswith('ss'):
        pass
    elif ing.endswith('s') and not ing.endswith('us'):
        ing = ing[:-1]

    # Final cleanup of non-letter chars at start/end
    ing = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', ing)
    
    return ing

def parse_txt_ingredients(file_path):
    """
    Parses ingredients from the specific text file format.
    Returns a list of lists (ingredients per recipe).
    """
    print(f"Parsing Text File: {file_path}")
    ingredients_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract content between ingredients: and [/INST]
                match = re.search(r'ingredients:\s*(.*?)\s*\[/INST\]', line)
                if match:
                    content = match.group(1).strip()
                    if not content:
                        ingredients_list.append([])
                        continue
                    
                    # Heuristic split by comma followed by digit
                    items = re.split(r', (?=\d)', content)
                    # Clean and lowercase
                    items = [item.strip().lower() for item in items if item.strip()]
                    ingredients_list.append(items)
                else:
                    ingredients_list.append([])
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return ingredients_list

def extract_quantity(text):
    """
    Extracts the first numeric quantity found in the ingredient text.
    Returns float, or 1.0 if no number is found.
    """
    match = re.search(r'(\d+(\.\d+)?)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 1.0

def extract_unit(text):
    """
    Finds the first matching unit in the text.
    Returns the unit string or 'unknown' if not found.
    """
    text_lower = text.lower()
    for unit in UNITS:
        pattern = r'\b' + re.escape(unit) + r'(?:es|s)?\b'
        if re.search(pattern, text_lower):
            return unit
    return "unknown"

# -------------------------------------------------------------------------
# EVALUATION METRICS
# -------------------------------------------------------------------------

def check_protein_sugar_tolerance(true_value, predicted_value):
    d_i = abs(true_value - predicted_value)
    if true_value <= 10:
        return d_i <= 2
    elif true_value <= 40:
        return d_i <= 0.2 * true_value
    else:
        return d_i <= 8

def check_fat_tolerance(true_value, predicted_value):
    d_i = abs(true_value - predicted_value)
    if true_value <= 10:
        return d_i <= 1.5
    elif true_value <= 40:
        return d_i <= 0.2 * true_value
    else:
        return d_i <= 8

def check_saturates_tolerance(true_value, predicted_value):
    d_i = abs(true_value - predicted_value)
    if true_value < 4:
        return d_i <= 0.8
    else:
        return d_i <= 0.2 * true_value

def evaluate_accuracy(y_true, y_pred, feature_name):
    """Calculates accuracy based on custom tolerance rules.
    Expects y_true and y_pred to be indexable (lists or arrays).
    """
    correct_count = 0
    total_count = len(y_true)
    
    for i in range(total_count):
        true_val = y_true[i]
        pred_val = y_pred[i]
        
        is_correct = False
        if feature_name in ['protein_g', 'sugars_g']:
            is_correct = check_protein_sugar_tolerance(true_val, pred_val)
        elif feature_name == 'fat_g':
            is_correct = check_fat_tolerance(true_val, pred_val)
        elif feature_name == 'saturates_g':
            is_correct = check_saturates_tolerance(true_val, pred_val)
        else:
            continue 
            
        if is_correct:
            correct_count += 1
            
    return correct_count / total_count if total_count > 0 else 0
