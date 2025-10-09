import json
import pandas as pd

# Load dataset to get actual relation names
df = pd.read_csv("dataset.csv")
relations = df["relation"].unique().tolist()

# Load config.json
with open("./model/config.json", "r") as f:
    config = json.load(f)

# Add proper mappings
config["id2label"] = {str(i): rel for i, rel in enumerate(relations)}
config["label2id"] = {rel: i for i, rel in enumerate(relations)}

# Save back
with open("./model/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Updated config.json with real relation labels!")
