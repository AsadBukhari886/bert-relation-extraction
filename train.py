import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Load data
df = pd.read_csv("dataset.csv")

# Prepare text with entity markers
df["text"] = df.apply(lambda r:
    r["sentence"].replace(r["entity1"], f"[E1] {r['entity1']} [/E1]")
                 .replace(r["entity2"], f"[E2] {r['entity2']} [/E2]"), axis=1)

relations = df["relation"].unique().tolist()
label2id = {r: i for i, r in enumerate(relations)}
id2label = {i: r for r, i in label2id.items()}
df["label"] = df["relation"].map(label2id)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

class RelationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(text, truncation=True, padding="max_length",
                                max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = RelationDataset(train_df["text"], train_df["label"], tokenizer)
test_dataset = RelationDataset(test_df["text"], test_df["label"], tokenizer)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(relations)
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete! Model saved in ./model/")
