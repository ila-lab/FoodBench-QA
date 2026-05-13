
import json
import random
import os

def split_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(BASE_DIR, "dataset_training_nutrient_estimation_deduplicated.json")
    txt_path = os.path.join(BASE_DIR, "dataset_training_nutrient_estimation_deduplicated.txt")

    output_dir = BASE_DIR
    base_name = "dataset_training_nutrient_estimation_deduplicated"

    print("Loading data...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    with open(txt_path, 'r') as f:
        txt_lines = f.readlines()

    if len(json_data) != len(txt_lines):
        raise ValueError(f"Mismatch in lengths: JSON ({len(json_data)}) vs TXT ({len(txt_lines)})")

    print(f"Total entries: {len(json_data)}")

    # Combine to shuffle together
    combined = list(zip(json_data, txt_lines))
    
    # Shuffle
    random.seed(42)
    random.shuffle(combined)

    # Split
    split_idx = int(len(combined) * 0.8)
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    print(f"Train set: {len(train_data)} entries")
    print(f"Test set: {len(test_data)} entries")

    # Unzip
    train_json, train_txt = zip(*train_data)
    test_json, test_txt = zip(*test_data)

    # Save
    print("Saving files...")
    
    train_json_path = os.path.join(output_dir, f"{base_name}_train.json")
    test_json_path = os.path.join(output_dir, f"{base_name}_test.json")
    train_txt_path = os.path.join(output_dir, f"{base_name}_train.txt")
    test_txt_path = os.path.join(output_dir, f"{base_name}_test.txt")

    with open(train_json_path, 'w') as f:
        json.dump(train_json, f, indent=4)
    
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f, indent=4)

    with open(train_txt_path, 'w') as f:
        f.writelines(train_txt)

    with open(test_txt_path, 'w') as f:
        f.writelines(test_txt)

    print("Done!")

if __name__ == "__main__":
    split_dataset()
