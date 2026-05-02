# Heart Disease Classifier 🫀

> ⚠️ **Work in progress** — more classifiers and features are actively being added.

A machine learning project that compares multiple classification algorithms to predict heart disease from clinical data.

## Overview

This project trains and evaluates three models — **Random Forest**, **Logistic Regression**, and **SVM** — on the [UCI Heart Disease dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci), using hyperparameter tuning and cross-validation to find the best performer.

## Models

| Model | Notes |
|---|---|
| Random Forest | Tuned on tree depth, estimator count, and leaf/split thresholds |
| Logistic Regression | L1/L2 regularization via `ElasticNet`, wrapped in a `StandardScaler` pipeline |
| SVM | Linear and RBF kernels, tuned with `StandardScaler` pipeline |

## Project Structure

```
├── main.py              # Training and evaluation script
├── model_utils.py       # evaluate_model() helper (grid search, metrics, feature importance)
├── heart.csv            # Dataset
└── README.md
```

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/your-username/heart-disease-classifier.git
cd heart-disease-classifier
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn
```

**3. Run**
```bash
python main.py
```

## Dataset

The dataset (`heart.csv`) contains 14 clinical features including age, sex, chest pain type, resting blood pressure, cholesterol, and others. The `target` column indicates the presence (1) or absence (0) of heart disease.

> Make sure `heart.csv` is in the root directory before running. You can download it from [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci).

## Requirements

- Python 3.8+
- pandas
- scikit-learn

## License

MIT