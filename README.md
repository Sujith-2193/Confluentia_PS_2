# Query-to-Category Text Classification using BERT and Gradio

An NLP solution for multilingual query understanding in e-commerce.

---

## 🧾 Problem Statement & Business Impact

**Understanding the Problem**

Large e-commerce platforms receive millions of user queries daily. These queries may be in different languages, contain typos, or be phrased in various informal ways. Retailers use structured product hierarchies to organize their catalogs.

**Goal:**

- Automatically map a user query to its correct product category path.
- Predict whether the query is accurate for the given category (`label = 1`) or not (`label = 0`).

**Business Impact:**

- Improves search relevance and recommendation systems.
- Enhances analytics for category performance.
- Reduces manual tagging efforts.
- Enables better personalization and user experience.

---

## 🧠 Approach & Methodology

**Solution Overview**

**Step-by-step Pipeline:**

1. **Data Ingestion**  
   - CSV file with columns: `language`, `origin_query`, `category_path`, `(label)`  
   - Diverse queries across multiple languages.

2. **Multilingual Support**  
   - Used `bert-base-multilingual-uncased` (supports 100+ languages natively).  
   - Directly feed raw queries to BERT without translation or spell correction.

3. **Model Architecture**  
   - Fine-tuned a pretrained BERT model using Hugging Face Transformers.  
   - Input: `[origin_query] [SEP] [category_path]`  
   - Output: Binary classification (`0` or `1`)  
   - Training using PyTorch with `CrossEntropyLoss`.

4. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score on a held-out validation set.

---

## 📊 Model Performance & Results

**Training Details:**

- Model: `bert-base-multilingual-uncased`
- Batch size: 16
- Epochs: 5
- Optimizer: `AdamW`
- Max sequence length: 128

**Performance:**

- Training loss improved from 0.60 → 0.11 over 5 epochs ✅
- Validation results:
  - Accuracy: 71.12%

**Observations:**

- Strong performance on relevant queries (class 1).  
- Class 0 is less balanced; may benefit from data augmentation or reweighting.

---

## 🧪 Deployment Interface using Gradio

**Why Gradio?**

- Instantly converts ML models into web apps.
- Supports CSV upload & download.
- Easy to run in Google Colab or locally.

**Interface Features:**

- Upload a CSV file with `origin_query` and `category_path`.
- Processes each row using the trained BERT model.
- Adds a new `prediction` column (0 or 1).
- Returns a downloadable CSV file with predictions.

**Colab Integration:**

- Gradio app runs in Colab using `interface.launch(share=True)`.
- Provides a public link without any hosting setup.

---

## ✅ Features

- Multilingual query classification
- Binary relevance prediction (0/1)
- CSV batch processing
- Downloadable prediction output
- Easy deployment using Gradio

---

## 🔧 Setup Steps

1. Clone the repository or download the code.
2. Install dependencies:

```bash
pip install torch transformers gradio pandas
