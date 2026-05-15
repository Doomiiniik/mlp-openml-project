# Letter Classification with MLP — End‑to‑End Machine Learning Pipeline

A complete, modular machine learning pipeline for **multiclass letter classification (A–Z)** using a **Multilayer Perceptron (MLP)** trained on the OpenML Letter Recognition dataset.  
The project demonstrates **production‑grade ML engineering practices**, including reproducible preprocessing, hyperparameter optimization, modular training, clean inference API, evaluation, and error analysis.

This repository is designed as a **portfolio‑ready ML project** showcasing engineering skills relevant for ML Engineer / Data Scientist roles.

---

## 🚀 Key Features

- **End‑to‑end ML pipeline** (preprocessing → tuning → training → inference → evaluation)
- **Optuna hyperparameter optimization** with 5‑fold cross‑validation
- **Config‑driven architecture** (`configs/`)
- **Modular codebase** (`src/`)
- **Reproducible CLI scripts** (`scripts/`)
- **Production‑style model saving & loading**
- **Comprehensive evaluation** (F1, confusion matrix, hardest classes)
- **Model card included** (`reports/model_card.md`)
- **Clean, industry‑standard project structure**

---

## 📊 Model Performance

**Dataset:** OpenML Letter Recognition 
**Classes:** 26 (A–Z)
**Test samples:** 3,734

| Metric | Score |
|--------|--------|
| **Accuracy** | ~96% |
| **Weighted F1** | ~96% |
| **Error Rate** | 4.04% |

### Most challenging classes  
The model shows lower performance on letters with similar geometric profiles:

- I, H, V, B, R

### Common confusions  
- G ↔ E 
- B ↔ V 
- R ↔ B 
- D ↔ O 
- I ↔ J 

All confusion patterns are **interpretable** and consistent with feature similarity.

---

## 🧠 Model Architecture

A fully‑connected **MLP classifier** with:

- 2–3 hidden layers 
- ReLU activations 
- Dropout regularization 
- Adam optimizer 
- Early stopping 

The exact architecture and hyperparameters are stored in:

configs/best_config.json


---

## 🛠 Project Structure
```
mlp-openml-project/
├── configs/              # Hyperparameter configs, Optuna results
├── data/                 # Processed datasets (train/test/val)
├── models/               # Final trained model
├── notebooks/            # EDA
├── outputs/              # Predictions, plots, error analysis
├── reports/              # Model card
├── scripts/              # CLI scripts for training/inference
├── run_pipeline.py       # Full training + evaluation pipeline
└── requirements.txt
```

This structure mirrors **real ML production repositories**.

---

## 🔧 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

python run_pipeline.py --all
```
or choose steps:
 ```
--preprocess
--tune
--train
--infer
--error
-viz
```
All outputs will appear in outputs/


📁 Artifacts

    models/best_model.pt — final trained MLP

    configs/best_config.json — best hyperparameters

    outputs/predictions.csv — predictions on test set

    outputs/error_analysis.json — misclassification statistics

    outputs/plots/ — confusion matrix, F1 scores, hardest classes

    reports/model_card.md — full model documentation
⚠️ Limitations

    Works only on numerical tabular features, not images

    Not suitable for OCR or handwritten text

    Sensitive to out‑of‑distribution inputs

    MLP interpretability is limited

📘 Model Card

Full model card available in:
reports/model_card.md

👤 Author

Dominik 
Machine Learning Engineer — Portfolio Project
