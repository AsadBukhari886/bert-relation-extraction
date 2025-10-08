import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

model_path = "./model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

df = pd.read_csv("dataset.csv")

relation_to_id = {
    "used_weapon": 0,
    "occurred_in": 1,
    "handled_by": 2,
    "targeted_location": 3,
    "injured_victims": 4,
    "killed_victims": 5,
    "motivated_by": 6,
    "arrested_by": 7,
    "suspect_of": 8,
    "mental_health_issue": 9
}
id_to_relation = {v: k for k, v in relation_to_id.items()}

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

preds, true_labels = [], []

print("🔍 Evaluating model on validation set...")

for _, row in val_df.iterrows():
    sentence = row["sentence"]
    entity1 = row["entity1"]
    entity2 = row["entity2"]
    relation_label = row["relation"]

    text = sentence.replace(entity1, f"[E1] {entity1} [/E1]").replace(entity2, f"[E2] {entity2} [/E2]")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()

    preds.append(pred_id)
    true_labels.append(relation_to_id[relation_label])

labels_used = sorted(unique_labels(true_labels + preds))
used_names = [list(relation_to_id.keys())[i] for i in labels_used]

report = classification_report(true_labels, preds, labels=labels_used, target_names=used_names, digits=4)

print("\nClassification Report:")
print(report)

with open("metrics.txt", "w") as f:
    f.write(report)
print("Metrics saved to metrics.txt")
