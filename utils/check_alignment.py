
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def parse_txt_line(line):
    # Format: [INST] ... [/INST] ... energy - X, fat - Y ...
    # precise parsing might be tricky due to variations, but let's try to extract numbers
    try:
        parts = line.split('[/INST]')
        if len(parts) < 2:
            return None
        return parts[1]
    except:
        return None

def extract_numbers(text):
    return re.findall(r"[\d\.]+", text)

def check_alignment():
    json_path = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_deduplicated.json")
    txt_path = os.path.join(DATA_DIR, "dataset_training_nutrient_estimation_deduplicated.txt")

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    with open(txt_path, 'r') as f:
        txt_lines = f.readlines()

    print(f"JSON entries: {len(json_data)}")
    print(f"TXT lines: {len(txt_lines)}")

    mismatches = 0
    for i in range(min(5, len(json_data))): # Check first 5
        j_ans = json_data[i].get('answer', '')
        t_ans = parse_txt_line(txt_lines[i])
        
        print(f"--- Entry {i} ---")
        print(f"JSON: {j_ans[:100]}...")
        print(f"TXT : {t_ans[:100].strip()}..." if t_ans else "TXT Parse Error")
        
        # Simple heuristic: check if energy value matches
        j_energy = re.search(r"energy - ([\d\.]+)", j_ans)
        t_energy = re.search(r"energy - ([\d\.]+)", t_ans) if t_ans else None
        
        if j_energy and t_energy:
            if j_energy.group(1) == t_energy.group(1):
                 print("MATCH")
            else:
                 print(f"MISMATCH: {j_energy.group(1)} vs {t_energy.group(1)}")
                 mismatches += 1
        else:
            print("Could not extract energy for comparison")
            mismatches += 1

    if mismatches > 0:
        print(f"\nFound {mismatches} mismatches in first 5 entries.")
    else:
        print("\nFirst 5 entries appear to align.")

if __name__ == "__main__":
    check_alignment()
