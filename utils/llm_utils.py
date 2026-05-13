"""
Shared helpers for LLM/API nutrient-prediction scripts.

Contains:
- SYSTEM_PROMPT          : the role/instruction string sent to Gemini / Ollama
- parse_nutrient_string  : regex extractor for "Nutrient values per 100 g: ..." outputs
- parse_inst_lines       : truncate dataset txt lines after [/INST] (strip ground truth)
"""

import re


SYSTEM_PROMPT = """
        System Role:
        You are a specialized nutritional data analyst. Your task is to calculate the nutrient profile per 100g for recipes provided in [INST] format.

        Instructions:
        Unit Conversion: Convert all units (e.g., pounds, cups, tablespoons, ml) to grams (g) using standard conversion factors (e.g., 1 cup water ≈ 236.6g, 1 tablespoon butter ≈ 14.2g).
        Calculation: Sum the total weight and total nutrients of all ingredients, then normalize the values to a 100g portion.
        Output Format: You must only provide the final result in this specific format:
        Nutrient values per 100 g: fat - [value], protein - [value], saturates - [value], sugars - [value]
        """


def parse_nutrient_string(response_str):
    """Extract fat/protein/saturates/sugars values from a model response string.

    Looks for the last occurrence of the "Nutrient values per 100 g" marker and
    then parses each nutrient name followed by a number (tolerant of -, :, **, etc).
    Returns {fat_g, protein_g, saturates_g, sugars_g} for keys that were found.
    """
    try:
        response_str = response_str.strip()

        marker = "Nutrient values per 100 g"
        idx = response_str.lower().rfind(marker.lower())
        text = response_str[idx:] if idx != -1 else response_str

        def find_value(key):
            m = re.search(rf'{key}\b[^\d]*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            return float(m.group(1)) if m else None

        keys = [("fat", "fat_g"), ("protein", "protein_g"),
                ("saturates", "saturates_g"), ("sugars", "sugars_g")]
        nutrients = {}
        for src, dst in keys:
            v = find_value(src)
            if v is not None:
                nutrients[dst] = v
        return nutrients
    except Exception as e:
        print(f"Error parsing response: {response_str}, Error: {e}")
        return {}


def parse_inst_lines(txt_path):
    """Read a dataset txt and keep only the [INST]...[/INST] prefix per line."""
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "[/INST]" in line:
                line = line.split("[/INST]")[0] + "[/INST]"
            lines.append(line)
    return lines
