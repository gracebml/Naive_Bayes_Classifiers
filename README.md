# Spam Classification with Multinomial Naive Bayes (Enron Spam Dataset)

This project implements a **spam email classifier** using **Multinomial Naive Bayes** with **Bag-of-Words (1-2 grams)** text representation. It includes data preprocessing, vectorization, model training, and evaluation using traditional NLP techniques.

---

## 📂 Project Structure
├── naive_bayes_classifiers.py # Multinomial Naive Bayes implementation (NumPy / SciPy)
├── process_data.py # Data cleaning + BOW vectorizer
├── requirements.txt # Required dependencies
├── submit.ipynb # Training notebook
└── Report.pdf # Project report


---

## Features

- Custom **Multinomial Naive Bayes** with Laplace smoothing
- Efficient sparse matrix representation
- Text preprocessing: cleaning, normalization
- Bag-of-Words with **n-grams (1,2)**
- Evaluation metrics: Accuracy, Precision, Recall, F1

## Model Details

- Smoothing: Laplace (alpha=1.0)

- Vectorization: Bag-of-Words with bigrams

- Stopwords removal enabled

- Sparse matrix optimization

- Log probability to prevent underflow

