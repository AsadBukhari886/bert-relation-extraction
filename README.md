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

## Output

<img width="1920" height="975" alt="relation_graph png" src="https://github.com/user-attachments/assets/bf8af9b7-aac2-4bdf-b2be-7354bb151f4a" />

---

## Project Structure

```bash
bert_relation_extraction_demo/
├── dataset.csv              # Input dataset: sentence, entity1, entity2, relation
├── train.py                 # Fine-tunes BERT model on relation extraction task
├── predict.py               # Interactive CLI for making predictions
├── visualize.py             # Graph visualization of predicted relations
├── evaluate.py              # Computes precision, recall, and F1-score
├── requirements.txt         # Python dependencies
├── .gitignore               # Files excluded from GitHub
├── model/                   # Saved trained model (excluded from repo)
├── results/                 # Training results (optional)
├── metrics.txt              # Evaluation results summary
└── relation_graph.png       # Visualization output image
```

---

## 🚀 Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py
# This trains the BERT model and saves it in the /model folder.
```

### Fix Config.py

```bash
python fix.config.py
# Run this command to fix config.json in ./model
```

### 4️⃣ Make Predictions
```bash
python predict.py
```
You can then interactively enter:
```bash
Sentence: John Doe used a rifle at a school.
Entity 1: John Doe
Entity 2: rifle
Output:
Predicted Relation: used_weapon
```

### 5️⃣ Visualization
```bash
python visualize.py
# This generates a graph showing all extracted relations.
```

---

## Example Visualization Output

*(Generated using NetworkX + Matplotlib)*

### Evaluation Results (sample run)

| Relation | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| used_weapon | 1.00 | 1.00 | 1.00 | 6 |
| occurred_in | 1.00 | 1.00 | 1.00 | 15 |
| handled_by | 1.00 | 1.00 | 1.00 | 9 |
| targeted_location | 1.00 | 1.00 | 1.00 | 10 |

**Accuracy:** 100.00%  
**Macro Avg F1:** 1.0000  
**Weighted Avg F1:** 1.0000  

🧩 **Note:** These results are from the fine-tuned BERT model trained on a well-structured synthetic dataset with clear relation boundaries.  
While performance is perfect on this dataset, additional testing on real-world and unseen examples is recommended to confirm generalization capability.

---

## 🎯 Future Improvements

- Increase dataset size for underrepresented relations  
- Experiment with BERT variants (e.g., roberta-base, bert-large)  
- Add confusion matrix visualization for clearer performance insight  
- Integrate Knowledge Graphs for advanced entity relation mapping  
- Implement RAG (Retrieval-Augmented Generation) to combine structured + unstructured reasoning  

---

## 👨‍💻 Author

**Asad Bukhari**  
Full Stack Developer & AI Research Enthusiast  

💼 Octek | AI + NLP + RAG Pipeline Developer  
🌍 Passionate about NLP, Relation Extraction & Knowledge Graphs  
📧 Contact: asadbukhari612@gmail.com
