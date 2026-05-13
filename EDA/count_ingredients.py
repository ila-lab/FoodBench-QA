import collections
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from nutrient_utils import normalize_ingredient, parse_txt_ingredients


def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'dataset_training_nutrient_estimation_deduplicated.txt'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    ingredients_per_recipe = parse_txt_ingredients(file_path)

    all_ings = [ing for sublist in ingredients_per_recipe for ing in sublist]
    print(f"Total raw ingredient occurrences: {len(all_ings)}")

    print("Generating normalization debug sample...")
    debug_data = []
    for raw in all_ings:
        norm = normalize_ingredient(raw)
        debug_data.append({'Raw': raw, 'Normalized': norm})
    
    # Sort by Normalized to group identical results
    debug_data.sort(key=lambda x: (x['Normalized'], x['Raw']))
        
    with open('normalization_debug.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['Raw', 'Normalized'])
        writer.writeheader()
        writer.writerows(debug_data)
        
    # Process all for counts
    cleaned_ings = []
    for ing in all_ings:
        cleaned = normalize_ingredient(ing)
        if cleaned:
            cleaned_ings.append(cleaned)
            
    clean_counter = collections.Counter(cleaned_ings)
    print(f"Total Unique Ingredients (After Normalization): {len(clean_counter)}")
    
    # Save all unique ingredients
    print("Saving unique ingredients list...")
    with open('unique_ingredients_list.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Ingredient', 'Count'])
        # Sort alphabetically by ingredient name
        sorted_ingredients = sorted(clean_counter.items(), key=lambda x: x[0])
        for ing, count in sorted_ingredients:
            writer.writerow([ing, count])

    print("\nTop 20 Ingredients:")
    for ing, count in clean_counter.most_common(20):
        print(f"{ing}: {count}")
        
    print(f"\nVerification files generated:\n1. unique_ingredients_list.csv (Full list)\n2. normalization_debug.csv (Random sample of {len(debug_data)} items)")

if __name__ == "__main__":
    main()
