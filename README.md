⭐ Automated Review Rating System

## 📖 Project Overview

The **Automated Review Rating System** is an **AI-driven deep learning framework** designed to predict **numerical ratings (1–5 stars)** directly from **textual user reviews**. The system automates sentiment-based rating generation by leveraging **Natural Language Processing (NLP)** and **Machine Learning (ML)** pipelines, enabling consistent and unbiased evaluation of customer feedback for products, services, or businesses.

The solution focuses on **text representation learning**, **feature extraction**, and **supervised model training** to capture semantic nuances within review texts and map them to corresponding star ratings.

---

## ✨ Key Features

* 🧠 **Automated Rating Prediction:** Infers the most probable star rating from free-form textual reviews.
* 🧹 **Text Preprocessing Pipeline:** Includes cleaning, tokenization, lemmatization, stopword removal, and vectorization.
* 🔤 **Feature Engineering:** Utilizes TF-IDF or word embeddings (Word2Vec / GloVe) to represent semantic structures.
* 📊 **Model Evaluation Framework:** Implements accuracy, RMSE, and F1-score for robust performance analysis.
* 🔄 **Modular Integration:** Supports real-time inference through serialized model deployment (using Pickle or H5 formats).

---

## 🗂 Dataset

### Sources

Datasets are aggregated from **Amazon**, **IMDB**, **TripAdvisor**, and similar e-commerce or review platforms.

### Structure

| Column   | Description                                                       |
| -------- | ----------------------------------------------------------------- |
| `Review` | User-generated text input describing a product or service.        |
| `Rating` | Integer label (1–5) indicating the customer’s satisfaction level. |

### Preprocessing Workflow

* ❌ **Noise Removal:** Eliminate HTML tags, punctuation, emojis, and special characters.
* 🔠 **Normalization:** Convert text to lowercase, remove stopwords, and apply stemming/lemmatization.
* 🔢 **Feature Transformation:** Apply TF-IDF vectorization or dense embeddings for text representation.
* ⚖️ **Class Rebalancing:** Use oversampling or class weights to handle skewed rating distributions.

---

## 🛠 Technology Stack

**Programming Language:** Python

**Core Libraries:**

* `pandas`, `numpy` → Data wrangling and numerical processing
* `scikit-learn` → Traditional ML algorithms and evaluation metrics
* `nltk`, `spaCy` → Text preprocessing and linguistic feature extraction
* `tensorflow` / `keras` → Deep learning model implementation (BiLSTM)
* `matplotlib`, `seaborn` → Exploratory Data Analysis (EDA) and visualization
* `pickle`, `joblib` → Model and tokenizer serialization for deployment

---

## 🚀 Installation & Setup

```bash
# Clone repository
git clone https://github.com/praveenk525/automated-review-rating-system.git  

# Navigate to directory
cd automated-review-rating-system  

# Install dependencies
pip install -r requirements.txt  

# Run the main script or Jupyter Notebook
python main.py
```

---

## 🖥 Usage

```python
import pickle
from tensorflow.keras.models import load_model

# Load trained tokenizer and label encoder
with open('tokenizer2.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder2.pkl', 'rb') as f:
    le = pickle.load(f)

# Load pre-trained model
model = load_model('Model_B.h5')

# Predict rating for a new review
review = ["This product exceeded my expectations!"]
predicted_rating = model.predict(review)
print(f"Predicted Rating: {predicted_rating[0]}")
```

---

## 🧩 Model Training Workflow

1. **Data Ingestion** → Load dataset and perform EDA.
2. **Text Preprocessing** → Tokenize, normalize, and vectorize review text.
3. **Train-Test Split** → Partition dataset for model generalization assessment.
4. **Feature Transformation** → Convert text into numerical vectors via TF-IDF or embeddings.
5. **Model Training**

   * Logistic Regression (Baseline)
   * BiLSTM (Deep Learning Architecture)
   * SVM (Optional high-margin classifier)
6. **Model Evaluation** → Compare models using metrics and cross-validation.
7. **Serialization** → Save trained model and preprocessing artifacts for deployment.

---

## 📊 Evaluation Metrics

| Metric                            | Description                                                   |
| --------------------------------- | ------------------------------------------------------------- |
| **Accuracy**                      | Fraction of correctly predicted ratings.                      |
| **RMSE (Root Mean Square Error)** | Quantifies deviation between predicted and actual scores.     |
| **Confusion Matrix**              | Displays class-level prediction distribution.                 |
| **F1-Score**                      | Harmonic mean of precision and recall for imbalanced classes. |

---

## 🔮 Future Enhancements

* 🤖 **Transformer-Based Models:** Fine-tune pre-trained architectures such as **BERT**, **RoBERTa**, or **DistilBERT** for contextual embedding.
* 🌐 **Multilingual Support:** Extend pipeline with multilingual tokenizers and translation layers.
* 📱 **Interactive Frontend:** Deploy through Flask/Django web interface or React-based UI for real-time prediction.
* 🧭 **Sentiment Analysis Integration:** Augment rating prediction with sentiment polarity and emotion detection for deeper insights.

