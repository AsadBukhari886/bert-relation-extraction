# import torch
# import json
# from transformers import BertTokenizer, BertForSequenceClassification

# # Load fine-tuned model and tokenizer
# model = BertForSequenceClassification.from_pretrained("./model")
# tokenizer = BertTokenizer.from_pretrained("./model")

# # Load id2label mapping from model config (if available)
# try:
#     with open("./model/config.json") as f:
#         config = json.load(f)
#     id2label = config.get("id2label", {})
# except Exception:
#     id2label = {}

# # Fallback for integer keys if needed
# id2label = {int(k): v for k, v in id2label.items()} if id2label else {0: "unknown"}

# # Prediction function
# def predict_relation(sentence, e1, e2):
#     """Predict relation between two entities."""
#     text = sentence.replace(e1, f"[E1] {e1} [/E1]").replace(e2, f"[E2] {e2} [/E2]")
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#         pred_id = torch.argmax(logits, dim=-1).item()
#     return id2label.get(pred_id, "unknown")

# # Interactive loop
# print("🧠 BERT Relation Extraction Demo (Type 'exit' to quit)\n")
# while True:
#     sentence = input("Enter a sentence: ")
#     if sentence.lower() == "exit":
#         break
#     e1 = input("Enter first entity: ")
#     e2 = input("Enter second entity: ")
#     relation = predict_relation(sentence, e1, e2)
#     print(f"\n🔹 Predicted Relation: {relation}\n{'-'*50}\n")

# # predict.py
# import torch
# import json
# import os
# from transformers import BertTokenizer, BertForSequenceClassification
# from visualize import visualize_relations_from_file  # 🔗 import the visualize function

# # Load model & tokenizer
# model = BertForSequenceClassification.from_pretrained("./model")
# tokenizer = BertTokenizer.from_pretrained("./model")

# # Load label mapping
# try:
#     with open("./model/id2label.json", "r") as f:
#         id2label = json.load(f)
#         id2label = {int(k): v for k, v in id2label.items()}
# except Exception:
#     id2label = {"0": "unknown"}

# def predict_relation(sentence, e1, e2):
#     text = sentence.replace(e1, f"[E1] {e1} [/E1]").replace(e2, f"[E2] {e2} [/E2]")
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#         pred_id = int(torch.argmax(logits, dim=-1).item())
#     return id2label.get(pred_id, f"unknown_{pred_id}")

# # Path to store all predictions
# relations_file = "relations.json"

# # Load existing predictions (if file exists)
# relations = []
# if os.path.exists(relations_file):
#     with open(relations_file, "r") as f:
#         relations = json.load(f)

# # Interactive prediction + auto-save
# print("🧠 Interactive Relation Extraction Demo (type 'exit' to quit)\n")
# while True:
#     s = input("Sentence: ")
#     if s.strip().lower() == "exit":
#         break
#     e1 = input("Entity 1: ")
#     e2 = input("Entity 2: ")

#     rel = predict_relation(s, e1, e2)
#     print(f"🔹 Predicted Relation: {rel}\n{'-'*50}")

#     # Save relation for visualization
#     relations.append({"entity1": e1, "relation": rel, "entity2": e2})
#     with open(relations_file, "w") as f:
#         json.dump(relations, f, indent=2)

# # After finishing predictions → visualize
# if relations:
#     print("\n📊 Generating visualization...")
#     visualize_relations_from_file(relations_file)
# else:
#     print("No relations predicted, nothing to visualize.")


# predict.py
import torch
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification
from visualize import visualize_relations_from_file

# Load model & tokenizer
model = BertForSequenceClassification.from_pretrained("./model")
tokenizer = BertTokenizer.from_pretrained("./model")

# Load id2label mapping robustly
id2label = {}

# First try id2label.json if exists
if os.path.exists("./model/id2label.json"):
    with open("./model/id2label.json", "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

# If not found, try reading from config.json
if not id2label:
    with open("./model/config.json", "r") as f:
        config = json.load(f)
        id2label = config.get("id2label", {})
        # Convert string keys -> int
        id2label = {int(k): v for k, v in id2label.items()}

print("✅ Loaded label mapping:", id2label)

def predict_relation(sentence, e1, e2):
    """Predict relation between two entities."""
    text = sentence.replace(e1, f"[E1] {e1} [/E1]").replace(e2, f"[E2] {e2} [/E2]")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())
    # Print raw id to confirm
    print(f"🔸 Model predicted class id: {pred_id}")
    return id2label.get(pred_id, f"unknown_{pred_id}")

# File to store all predicted relations
relations_file = "relations.json"

# Load existing predictions (if any)
relations = []
if os.path.exists(relations_file):
    with open(relations_file, "r") as f:
        relations = json.load(f)

print("\n🧠 Interactive Relation Extraction (type 'exit' to quit)\n")
while True:
    s = input("Sentence: ")
    if s.strip().lower() == "exit":
        break
    e1 = input("Entity 1: ")
    e2 = input("Entity 2: ")

    rel = predict_relation(s, e1, e2)
    print(f"🔹 Predicted Relation: {rel}\n{'-'*50}")

    relations.append({"entity1": e1, "relation": rel, "entity2": e2})
    with open(relations_file, "w") as f:
        json.dump(relations, f, indent=2)

# Visualize after exiting
if relations:
    print("\n📊 Generating visualization...")
    visualize_relations_from_file(relations_file)
else:
    print("No relations predicted, nothing to visualize.")
