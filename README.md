# BERT Relation Extraction Demo

This project demonstrates **relation extraction** using a fine-tuned **BERT** model.  
It identifies semantic relationships between entities (e.g., *used_weapon*, *occurred_in*, *handled_by*) from incident-related text data.

---

## Overview

The goal of this project is to extract meaningful **entity relationships** from text, similar to how research papers convert unstructured data (e.g., mass shooting reports) into structured knowledge graphs.

### Example

**Input Sentence:**
> John Doe used a rifle at a school.

**Predicted Output:**
> `John Doe — used_weapon → rifle`

---

## Tech Stack

| Component | Description |
|------------|-------------|
| **Python** | Core programming language |
| **PyTorch** | Deep learning backend |
| **Hugging Face Transformers** | BERT fine-tuning & tokenization |
| **Pandas / scikit-learn** | Data preprocessing & evaluation metrics |
| **Matplotlib / NetworkX** | Visualization of extracted relations |

---

## OutPut


---

## Project Structure

```bash
bert_relation_extraction_demo/
│
├── dataset.csv              # Input dataset: sentence, entity1, entity2, relation
├── train.py                 # Fine-tunes BERT model on relation extraction task
├── predict.py               # Interactive CLI for making predictions
├── visualize.py             # Graph visualization of predicted relations
├── evaluate.py              # Computes precision, recall, and F1-score
│
├── requirements.txt         # Python dependencies
├── .gitignore               # Files excluded from GitHub
│
├── model/                   # Saved trained model (excluded from repo)
├── results/                 # Training results (optional)
│
├── metrics.txt              # Evaluation results summary
└── relation_graph.png       # Visualization output image


---

## 🚀 Setup Instructions

## 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows


# OR
source venv/bin/activate     # macOS/Linux
---

## 2️⃣ Install Dependencies

pip install -r requirements.txt

---
## 3️⃣ Train the Model

python train.py
This trains the BERT model and saves it in the /model folder.
---
##4️⃣ Make Predictions

python predict.py

You can then interactively enter:

Sentence: John Doe used a rifle at a school.
Entity 1: John Doe
Entity 2: rifle
Output:

 Predicted Relation: used_weapon
---
## Visualization
After making a few predictions, run:

python visualize.py
This generates a graph showing all extracted relations.

---
## Example Visualization Output:


(Generated using NetworkX + Matplotlib)

Evaluation Results
Validation Results (sample run)

Relation	Precision	Recall	F1-score	Support
used_weapon	0.44	1.00	0.61	4
occurred_in	0.33	1.00	0.50	1
targeted_location	0.00	0.00	0.00	2
killed_victims	0.00	0.00	0.00	1
motivated_by	0.00	0.00	0.00	1
arrested_by	0.00	0.00	0.00	1
mental_health_issue	0.00	0.00	0.00	3

Accuracy: 38.46%
Macro Avg F1: 0.1593
Weighted Avg F1: 0.2278

🧩 Note: These results are from an early prototype trained on a small, imbalanced dataset.
With more balanced data and 5–8 epochs of training, the model performance is expected to improve significantly.

🎯 Future Improvements
📈 Increase dataset size for underrepresented relations

🧩 Experiment with BERT variants (e.g., roberta-base, bert-large)

🧠 Add confusion matrix visualization for clearer performance insight

🔗 Integrate Knowledge Graphs for advanced entity relation mapping

🚀 Implement RAG (Retrieval-Augmented Generation) to combine structured + unstructured reasoning

👨‍💻 Author
Asad Bukhari
Full Stack Developer & AI Research Enthusiast

💼 Octek | AI + NLP + RAG Pipeline Developer

🌍 Passionate about NLP, Relation Extraction & Knowledge Graphs

📧 Contact: asadbukhari612@gmail.com