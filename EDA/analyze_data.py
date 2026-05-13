import json
import re
import os
import sys
import collections
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def parse_txt_ingredients(file_path):
    print(f"Parsing Text File: {file_path}")
    ingredients_list = []
    
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
    
    return ingredients_list

def load_data(txt_path, json_path):
    print("Loading data...")
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Load Ingredients
    ingredients_per_recipe = parse_txt_ingredients(txt_path)

    # Combine into DataFrame
    data_records = []
    for i, entry in enumerate(json_data):
        rec = {'id': i}
        
        # Nutrients
        if 'nutrients_per_100g' in entry and entry['nutrients_per_100g']:
            rec.update(entry['nutrients_per_100g'])
        
        # Ingredients
        ing_list = ingredients_per_recipe[i]
        rec['ingredient_count'] = len(ing_list)
        rec['raw_ingredients'] = ing_list # Store list for now
        
        data_records.append(rec)
        
    df = pd.DataFrame(data_records)
    
    # Fill missing numeric values with 0 or drop? Let's drop rows with completely missing nutrient data for analysis
    # But keep them if we just analyze ingredients.
    # For correlation/clustering, we need numeric data.
    return df

def normalize_ingredient(ing, remove_units=True):
    # Lowercase and strip
    ing = ing.strip().lower()
    
    # 1. Remove parenthesized content FIRST (e.g. "1 (15 oz) can") -> "1  can"
    ing = re.sub(r'\(.*?\)', '', ing)

    # 1b. Remove "number to number" patterns e.g. "1 to 2"
    ing = re.sub(r'\d+\s*to\s*\d+', '', ing)
    
    # 2. Remove leading specific characters/numbers
    # e.g. "1", "1/2", "1-2", "1.5", "*"
    ing = re.sub(r'^[\d\s/\.\-\*]+', '', ing).strip()
    
    if remove_units:
        # define units ONLY (No adjectives)
        units = [
            'bushel', 'shot', 'glass', 'cup', 'tablespoon', 'tbsp', 'tsp', 'kg', 'teaspoon', 
            'g', 'ml', 'ounce', 'oz', 'pound', 'lb', 'quart', 'pint', 'gallon', 'dash', 'drop', 
            'pinch', 'fl. oz', 'scoop', 'liter', 'l', 'can', 'bottle', 'package', 'stick', 'slice', 
            'clove', 'head', 'stalk', 'sprig', 'bunch', 'piece', 'container', 'jar', 'bag', 'box', 'envelope'
        ]
        
        words_to_remove = units
        # Sort long to short
        words_to_remove.sort(key=len, reverse=True)
        
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

def analyze_top_ingredients(df, output_dir):
    print("Analyzing Ingredient Frequencies...")
    all_ings = [ing for sublist in df['raw_ingredients'] for ing in sublist]
    
    # TEST: Count with units preserved (remove_units=False)
    raw_ings_normalized_with_units = []
    for ing in all_ings:
        cleaned = normalize_ingredient(ing, remove_units=False)
        if cleaned:
            raw_ings_normalized_with_units.append(cleaned)
    
    clean_counter_with_units = collections.Counter(raw_ings_normalized_with_units)
    print(f"\n[TEST] Total Unique Ingredients (Keeping Quantifiers/Units): {len(clean_counter_with_units)}")
    
    # Plot Top 20 with units
    top_20_with_units = clean_counter_with_units.most_common(20)
    # print("\nTop 20 Ingredients (Keeping Quantifiers/Units):")
    # for ing, count in top_20_with_units:
    #     print(f"  {ing}: {count}")

    plt.figure(figsize=(12, 8))
    sns.barplot(x=[x[1] for x in top_20_with_units], y=[x[0] for x in top_20_with_units])
    plt.title("Top 20 Most Frequent Ingredients (Keeping Quantifiers/Units)")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_20_ingredients_with_units.png"))
    plt.close()

    # Clean ingredients using the new normalization function (Default: remove_units=True)
    cleaned_ings = []
    for ing in all_ings:
        cleaned = normalize_ingredient(ing)
        if cleaned: # Only add if not empty
            cleaned_ings.append(cleaned)
        
    clean_counter = collections.Counter(cleaned_ings)
    print(f"Total Unique Ingredients (After Normalization & Unit Removal): {len(clean_counter)}")
    top_50 = [ing for ing, count in clean_counter.most_common(50)]
    
    # Plot Top 20
    top_20 = clean_counter.most_common(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[x[1] for x in top_20], y=[x[0] for x in top_20])
    plt.title("Top 20 Most Frequent Ingredients (Normalized)")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_20_ingredients.png"))
    plt.close()
    
    return top_50

def add_ingredient_features(df, top_ingredients):
    # Add binary columns for top ingredients
    for ing_name in top_ingredients:
        col_name = f"has_{ing_name.replace(' ', '_')}"
        safe_ing = re.escape(ing_name)
        df[col_name] = df['raw_ingredients'].apply(lambda x: 1 if any(safe_ing in s for s in x) else 0)
    return df

def analyze_correlations(df, output_dir):
    print("Analyzing Correlations...")
    # Select numeric columns
    nutrients = ['fat_g', 'protein_g', 'saturates_g', 'sugars_g', 'ingredient_count']
    sub_df = df[nutrients].dropna()
    corr = sub_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Nutrient Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nutrient_correlation_matrix.png"))
    plt.close()

def analyze_ingredient_nutrient_correlation(df, top_ingredients, output_dir):
    print("Analyzing Ingredient-Nutrient Impacts...")
    nutrients = ['energy_kcal', 'fat_g', 'protein_g', 'salt_g', 'sugars_g']
    
    # Calculate correlation between "has_ingredient" binary cols and nutrients
    impact_data = []
    
    for ing in top_ingredients:
        col_name = f"has_{ing.replace(' ', '_')}"
        if col_name not in df.columns: continue
        
        row = {'Ingredient': ing}
        for nutri in nutrients:
            if nutri not in df.columns: continue
            # Correlation
            corr = df[col_name].corr(df[nutri])
            row[nutri] = corr
        impact_data.append(row)
        
    impact_df = pd.DataFrame(impact_data).set_index('Ingredient')
    
    # Heatmap of impacts
    plt.figure(figsize=(12, 10))
    sns.heatmap(impact_df, cmap='RdBu_r', center=0, annot=False) # Annot false primarily due to size
    plt.title("Impact of Top Ingredients on Nutrients (Correlation)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ingredient_nutrient_impact.png"))
    plt.close()
    
    # Print top positive correlations for each nutrient
    print("\nTop Ingredient Drivers (Highest Correlation):")
    for nutri in nutrients:
        top = impact_df[nutri].sort_values(ascending=False).head(3)
        print(f"  {nutri}: {', '.join([f'{i} ({c:.2f})' for i, c in top.items()])}")

def detect_outliers(df, output_dir):
    print("Detecting Outliers...")
    nutrients = ['energy_kcal', 'salt_g', 'sugars_g']
    sub_df = df[nutrients].dropna()
    
    # Z-score > 3
    z_scores = np.abs(zscore(sub_df))
    outliers = (z_scores > 4).any(axis=1) # Using 4 sigma for really extreme values
    
    outlier_rows = sub_df[outliers]
    print(f"\nFound {len(outlier_rows)} potential outlier recipes (Values > 4 Std Dev).")
    
    if not outlier_rows.empty:
        print("Top 5 Extreme Recipes (by Energy):")
        print(outlier_rows.sort_values('energy_kcal', ascending=False).head(5))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_file = os.path.join(base_dir, 'dataset_training_nutrient_estimation_deduplicated.txt')
    json_file = os.path.join(base_dir, 'dataset_training_nutrient_estimation_deduplicated.json')
    output_dir = os.path.join(base_dir, 'analysis_results')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    df = load_data(txt_file, json_file)
    
    if df.empty:
        print("No data loaded.")
        return

    # 2. Basic Stats & Visualizations
    # (Previously implemented histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ingredient_count'], bins=max(df['ingredient_count']))
    plt.title('Distribution of Ingredients per Recipe')
    plt.savefig(os.path.join(output_dir, 'ingredients_per_recipe_hist.png'))
    plt.close()

    # 3. Top Ingredients
    top_50 = analyze_top_ingredients(df, output_dir)
    
    # 4. Feature Engineering
    df = add_ingredient_features(df, top_50)
    
    # 5. Correlations
    analyze_correlations(df, output_dir)
    
    # 7. Outliers
    detect_outliers(df, output_dir)
    
    print(f"\nAnalysis Complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
