# Employee-Turnover-Vacancy-Analytics-with-NLP-Python-PyTorch-Hugging-Face-scikit-learn


Classifies Indonesian exit-interview text into resignation reason categories (**R / UR**) using a fine-tuned multilingual DistilBERT, then combines the predictions with turnover and vacancy data to find where and why people leave.

> **Data privacy:** The source data (HR resignation records) contains employee names, IDs, and other personal information. **No raw data is included in this repository.** Only code and aggregated results are shared.

## Overview

| | |
|---|---|
| **Task** | Binary text classification of exit-interview reasons (R / UR) |
| **Language** | Indonesian (informal, with HR abbreviations) |
| **Model** | `distilbert-base-multilingual-cased` (Hugging Face Transformers) |
| **Imbalance handling** | SMOTE (imbalanced-learn) |
| **Data period** | Jul 2024 – Apr 2025, 1,400+ resignation records, 6 regions |
| **Validation macro-F1** | 0.79 (accuracy 0.79) |

## Workflow

1. **Cleaning** – lowercase, strip non-letters, expand HR shorthand (`eoc` → `end of contract`, `thd` → `terhadap`, `ybs` → `yang bersangkutan`).
2. **Labeling** – keep records with a valid `R` / `UR` category, encode with `LabelEncoder`.
3. **Balancing** – SMOTE on tokenized inputs to balance the two classes.
4. **Fine-tuning** – DistilBERT multilingual, 3 epochs, batch size 8, weight decay 0.01 (trained on CPU).
5. **Evaluation** – classification report on the validation split.
6. **Batch inference** – predict reasons for the full resignation file and export to Excel.
7. **Analysis & visualization** – word cloud, employment-status breakdown, treemap of absenteeism cases by division, average vacancy days before replacement per division.

## Results

Validation set (1,990 samples, balanced):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| R | 0.73 | 0.94 | 0.82 |
| UR | 0.91 | 0.65 | 0.76 |
| **Macro avg** | 0.82 | 0.79 | **0.79** |

Applied to 1,317 resignation records: 1,045 predicted positive, 272 predicted negative.

Example findings from the analysis:
- Absenteeism ("mangkir") among contract employees was heavily concentrated in a single division.
- Average vacancy days before a replacement joined varied widely by division.

## Limitations

- **Evaluation caveat:** SMOTE was applied *before* the train/validation split, so the validation set contains synthetic samples. The reported F1 is likely optimistic.
- SMOTE on token IDs interpolates between integer IDs, which has no real semantic meaning. Class weights, focal loss, or oversampling of raw text would be more principled.
- The model is weaker at recalling `UR` (0.65) than `R` (0.94).
- The model is trained on one company's HR data and may not generalize elsewhere.

## Planned improvements

- [ ] Split first, apply SMOTE (or class weights / focal loss) on the training set only, and evaluate on untouched real data
- [ ] Add cross-validation and a confusion matrix
- [ ] Compare against a TF-IDF + logistic regression baseline

## Tech stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · imbalanced-learn · pandas · matplotlib · seaborn · Plotly · WordCloud

## Repository structure

```
.
├── smote.ipynb      # preprocessing, SMOTE, training, evaluation, inference, analysis
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook smote.ipynb
```

You will need your own dataset with an exit-interview text column and an `R` / `UR` label column. Update the file paths in the notebook accordingly.

## Author

[Your Name](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)
