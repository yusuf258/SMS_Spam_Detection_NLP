# SMS Spam Detection | NLP Classification

NLP-based binary classification system to detect spam SMS messages using both classical machine learning and deep learning approaches.

## Problem Statement
Automatically classify incoming SMS messages as **Spam** or **Ham (legitimate)** to protect users from unwanted and potentially malicious content.

## Dataset
| Attribute | Detail |
|---|---|
| File | SMS Spam Collection dataset |
| Records | ~5,572 messages |
| Classes | Ham (legitimate) / Spam |
| Class Balance | ~87% Ham / ~13% Spam (imbalanced) |

## Methodology
1. **EDA & Visualization** — Message length distribution, word clouds, class balance analysis
2. **Text Preprocessing** — Lowercasing, punctuation removal, stopword filtering, tokenization
3. **Feature Extraction** — TF-IDF vectorization
4. **ML Models** — Naive Bayes, Logistic Regression, SVM (comparison)
5. **DL Model** — Embedding + LSTM / Dense layers
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Results
| Model | Accuracy |
|---|---|
| **SVM (Best ML)** | **97.94%** |
| Deep Learning (DL) | 97.22% |

> SVM with TF-IDF features achieves near state-of-the-art performance on this dataset.

## Technologies
`Python` · `scikit-learn` · `TensorFlow/Keras` · `NLTK` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `joblib`

## File Structure
```
04_SMS_Spam_Detection_NLP/
├── project_notebook.ipynb   # Main notebook
├── spam.csv                 # Dataset
└── models/                  # Saved model files
```

## How to Run
```bash
cd 04_SMS_Spam_Detection_NLP
jupyter notebook project_notebook.ipynb
```
