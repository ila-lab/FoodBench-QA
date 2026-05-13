import re
import collections
import os
import sys
import csv
import json

def deduplicate_files(txt_path, json_path):
    print(f"Deduplicating files:\n  TXT: {txt_path}\n  JSON: {json_path}")
    
    if not os.path.exists(txt_path) or not os.path.exists(json_path):
        print("Error: Input files not found.")
        return

    # Read JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Read TXT lines
    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_lines = f.readlines()
        
    if len(json_data) != len(txt_lines):
        print(f"Error: Line count mismatch! JSON: {len(json_data)}, TXT: {len(txt_lines)}")
        # Proceeding might be dangerous if they are not aligned, but let's try to zip them effectively.
        # Ideally we should stop, but user might want best effort? 
        # For now, strict check.
        return

    seen_signatures = set()
    new_json_data = []
    new_txt_lines = []
    
    duplicates_count = 0
    kept_count = 0
    
    for i, (line, json_entry) in enumerate(zip(txt_lines, json_data)):
        line = line.strip()
        if not line:
            continue
            
        ing_sig, _, _ = parse_line(line)
        
        if ing_sig is not None:
            if ing_sig in seen_signatures:
                duplicates_count += 1
                continue
            
            seen_signatures.add(ing_sig)
            
            # Keep this entry
            new_txt_lines.append(line)
            
            # Update scan_id/case_id to be sequential
            # Assuming 'case_id' is the key based on file viewing
            json_entry['case_id'] = str(kept_count + 1)
            new_json_data.append(json_entry)
            
            kept_count += 1
        else:
            # If we fail to parse, do we keep it? 
            # Current logic in check_duplicates just counts them.
            # Let's keep existing potentially invalid lines if we can't parse them to be safe?
            # Or better, if we can't determine duplication, maybe we should keep it UNIQUE by line content?
            # Let's assume valid lines for now based on strict regex specific to this dataset.
            # If parse fails, fallback to line content as signature?
            print(f"Warning: Could not parse line {i+1}. Skipping deduplication check for this line and preserving it.")
            new_txt_lines.append(line)
            json_entry['case_id'] = str(kept_count + 1)
            new_json_data.append(json_entry)
            kept_count += 1

    # Define new filenames
    dir_name = os.path.dirname(txt_path)
    base_txt = os.path.basename(txt_path).rsplit('.', 1)[0]
    base_json = os.path.basename(json_path).rsplit('.', 1)[0]
    
    new_txt_path = os.path.join(dir_name, f"{base_txt}_deduplicated.txt")
    new_json_path = os.path.join(dir_name, f"{base_json}_deduplicated.json")
    
    # Write new files
    with open(new_txt_path, 'w', encoding='utf-8') as f:
        for line in new_txt_lines:
            f.write(line + '\n')
            
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, indent=2, ensure_ascii=False)
        
    print("-" * 30)
    print(f"Original Count: {len(txt_lines)}")
    print(f"Duplicates Removed: {duplicates_count}")
    print(f"New Count: {kept_count}")
    print("-" * 30)
    print(f"Created new files:")
    print(f"  TXT: {new_txt_path}")
    print(f"  JSON: {new_json_path}")

def parse_line(line):
    """
    Parses a line to extract the canonical ingredient signature and the parsed nutrient values.
    Uses 'energy' as an anchor to find the start of nutrients.
    """
    # Pattern to capture ingredients and the rest (nutrients)
    match = re.search(r'ingredients:\s*(.*?)\s*\[/INST\].*?(energy.*)', line)
    
    if match:
        ing_content = match.group(1).strip()
        raw_nutrient_str = match.group(2).strip()
        if not ing_content:
            return tuple(), {}
        
        # --- Parse Ingredients ---
        # Heuristic split for ingredients
        items = re.split(r', (?=\d)', ing_content)
        # Clean, lowercase, and strip ingredients
        items = [item.strip().lower() for item in items if item.strip()]
        # Canonical signature: Sorted tuple of ingredients
        ing_signature = tuple(sorted(items))
        
        # --- Parse Nutrients ---
        # Find the start of the nutrient data using specific keywords as anchors
        # The text usually starts with duplicates like "Nutrient values..." but always contains "energy -"
        
        # Locate "energy" or "energy -"
        # We lowercase just in case, though usually it is lowercase "energy"
        lower_nutrient_str = raw_nutrient_str.lower()
        start_idx = lower_nutrient_str.find('energy')
        
        if start_idx == -1:
            # Fallback: maybe it doesn't have energy? Return empty or try to parse whole string
            nutrient_data_str = raw_nutrient_str
        else:
            # Start parsing from 'energy'
            nutrient_data_str = raw_nutrient_str[start_idx:]
            
        nutrient_map = {}
        
        # Split by comma to get "key - value" pairs
        parts = nutrient_data_str.split(',')
        for part in parts:
            if ' - ' in part:
                 name, val = part.split(' - ', 1)
                 name = name.strip().lower()
                 
                 # Sometimes the last value might have extra text if we didn't slice correctly?
                 # content usually looks like: "energy - 100, fat - 10"
                 # val should be a number.
                 val = val.strip()
                 try:
                     val_float = float(val)
                     nutrient_map[name] = val_float
                 except ValueError:
                     pass 
        
        return ing_signature, nutrient_map, raw_nutrient_str
    
    return None, None, None

def check_duplicates(file_path):
    print(f"Checking for duplicates in {file_path}")
    print("Using 'energy' keyword to anchor nutrient parsing.")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    total_lines = 0
    # Map: Ingredient Tuple -> List of (Line Number, Nutrient Dict, Raw Nutrient Str)
    recipe_map = collections.defaultdict(list)
    failed_parse_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):

                line = line.strip()

                if not line:
                    continue
                
                total_lines += 1
                
                ing_sig, nutrient_dict, raw_n_str = parse_line(line)
                
                if ing_sig is not None:
                    recipe_map[ing_sig].append({
                        'line': line_num,
                        'nutrients': nutrient_dict,
                        'raw_nutrient_str': raw_n_str
                    })
                else:
                    failed_parse_count += 1

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Analyze
    unique_recipes = len(recipe_map)
    redundant_duplicates = 0
    inconsistent_records = 0
    
    # Prepare Report
    report_rows = []

    for ing_sig, occurrences in recipe_map.items():
        if len(occurrences) > 1:
            # Check consistency based on parsed nutrient values
            # We take the first occurrence as the 'reference'
            ref_nutrients = occurrences[0]['nutrients']
            
            # Helper: Equality with tolerance
            def are_nutrients_equal(n1, n2):
                if not n1 and not n2: return True # Both empty/failed parse
                if len(n1) != len(n2): return False
                for k, v in n1.items():
                    if k not in n2: return False
                    if abs(v - n2[k]) > 0.001: return False 
                return True

            is_consistent = all(are_nutrients_equal(occ['nutrients'], ref_nutrients) for occ in occurrences)
            
            duplicate_status = "Full Match" if is_consistent else "Conflicting Nutrients"
            
            if is_consistent:
                redundant_duplicates += (len(occurrences) - 1)
            else:
                inconsistent_records += 1
            
            # Add to report
            for occ in occurrences:
                # Format parsed nutrients tightly
                if occ['nutrients']:
                    nutr_str = "; ".join([f"{k}: {v}" for k,v in occ['nutrients'].items()])
                else:
                    nutr_str = f"FAILED_PARSE (Raw: {occ['raw_nutrient_str'][:50]}...)"
                
                report_rows.append({
                    'Line_Number': occ['line'],
                    'Duplicate_Status': duplicate_status,
                    'Ingredients': ', '.join(ing_sig),
                    'Nutrients_Parsed': nutr_str
                })

    # Sort report
    report_rows.sort(key=lambda x: (x['Ingredients'], x['Line_Number']))

    # Write CSV
    output_csv = 'duplicate_recipes_report.csv'
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['Line_Number', 'Duplicate_Status', 'Ingredients', 'Nutrients_Parsed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\n📄 CSV Report generated: {output_csv}")
        print(f"   Contains {len(report_rows)} rows.")
    except Exception as e:
        print(f"Error writing CSV: {e}")

    print("\n" + "="*60)
    print("RECIPE CONSISTENCY ANALYSIS (Energy Anchor)")
    print("="*60)
    print(f"Total Lines Processed: {total_lines}")
    print(f"Unique Ingredient Sets: {unique_recipes}")
    print("-" * 30)
    print(f"REDUNDANT DUPLICATES (Consistent): {redundant_duplicates}")
    print(f"INCONSISTENT RECORDS (Conflicting): {inconsistent_records}")
    print("="*60)
    
    if inconsistent_records > 0:
        print("\nTop 5 Inconsistencies:")
        count = 0
        for row in report_rows:
            if row['Duplicate_Status'] == "Conflicting Nutrients":
                 print(f"Line {row['Line_Number']}: {row['Nutrients_Parsed']}")
                 count += 1
                 if count >= 5: break

if __name__ == "__main__":
    # Default paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_file = os.path.join(base_dir, 'dataset_training_nutrient_estimation.txt')
    json_file = os.path.join(base_dir, 'dataset_training_nutrient_estimation.json')
    
    if len(sys.argv) > 1:
        txt_file = sys.argv[1]
    
    # Start deduplication
    deduplicate_files(txt_file, json_file)
