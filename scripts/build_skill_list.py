import json
from pathlib import Path

RAW_DIR = Path("data/skills")
OUTPUT_FILE = Path("data/skills/skill_list.json")

all_skills = set()

def extract_skills(data):
    """Recursively extract skill names from JSON."""
    if isinstance(data, list):
        for item in data:
            extract_skills(item)

    elif isinstance(data, dict):
        # if dict has 'label' key
        if "label" in data:
            label = data["label"]
            if isinstance(label, dict) and "en" in label:
                all_skills.add(label["en"].strip().lower())
            elif isinstance(label, str):
                all_skills.add(label.strip().lower())

        for value in data.values():
            extract_skills(value)

    elif isinstance(data, str):
        if 1 < len(data) < 50:
            all_skills.add(data.strip().lower())

# Debug: check which files are detected
files = list(RAW_DIR.glob("*.json"))
print(f"Found {len(files)} JSON files: {[f.name for f in files]}")

# Read all JSON files
for file in files:
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
            extract_skills(data)
    except Exception as e:
        print(f"Skipping {file}: {e}")

# Save cleaned skill list
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sorted(all_skills), f, indent=2)

print(f"Saved {len(all_skills)} skills to {OUTPUT_FILE}")
