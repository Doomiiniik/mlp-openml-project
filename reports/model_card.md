# Model Card — Letter Classification MLP

## 1. Model Overview
This model is a **Multilayer Perceptron (MLP)** designed to classify letters A–Z using numerical features from the OpenML Letter Recognition dataset.  
The model was optimized using **Optuna** and trained with the best hyperparameters discovered during tuning.  
It is part of a modular ML pipeline including preprocessing, hyperparameter search, final training, inference, error analysis, and visualization.

---

## 2. Intended Use

### Primary Use Cases
- Educational and demonstrational ML projects  
- Benchmarking MLP performance on tabular data  
- Portfolio / CV presentation  
- Research on classification performance across 26 balanced classes  

### Out-of-Scope Uses
The model is **not intended** for:
- Optical Character Recognition (OCR) on images  
- Handwritten character recognition  
- Any safety‑critical or real‑world production deployment  
- Processing sensitive or personal data  

---

## 3. Dataset
- Source: **OpenML Letter Recognition Dataset**  
- Classes: **26 letters (A–Z)**  
- Data type: **Tabular numerical features**  
- Balanced distribution across classes  
- No missing values  
- Test set size: **3,734 samples**

---

## 4. Training Procedure

### 4.1 Preprocessing
- Standardization of all numerical features  
- Label encoding of target classes (saved as `label_encoder.pkl`)  
- Deterministic train/validation/test split  
- All processed data stored in `data/processed/`

### 4.2 Hyperparameter Tuning
- Framework: **Optuna**  
- Search space included:
  - number of hidden layers  
  - neurons per layer  
  - dropout rate  
  - batch size  
  - learning rate  
- Validation: **5‑fold cross‑validation**  
- Best configuration saved as `configs/best_config.json`

### 4.3 Final Training
- Training on combined train + validation sets  
- Optimizer: **Adam**  
- Early stopping applied  
- Final model saved as `models/best_model.pt`

---

## 5. Evaluation

### 5.1 Test Metrics
On the held‑out test set:

- **Accuracy:** ~96%  
- **Weighted F1-score:** ~96%  
- **Error rate:** 4.04%  

### 5.2 Per-Class Performance
All classes achieve **F1 ≥ 0.93**, indicating stable and consistent performance.

**Most challenging classes:**

| Letter | F1-score |
|--------|----------|
| I | 0.932 |
| H | 0.9324 |
| V | 0.9424 |
| B | 0.9431 |
| R | 0.9435 |

### 5.3 Confusion Patterns
The model tends to confuse letters with similar feature profiles:

- G ↔ E
- B ↔ V
- R ↔ B
- D ↔ O
- I ↔ J

These are **systematic and interpretable** errors.

---

## 6. Limitations
- Works **only** on numerical tabular features, not images
- Not suitable for handwritten or stylized characters
- Sensitive to out-of-distribution inputs
- Limited interpretability (MLP is a black‑box model)
- Not designed for real‑world deployment

---

## 7. Ethical Considerations
- Dataset contains **no personal or sensitive data**
- No demographic attributes → no risk of bias or discrimination
- No privacy concerns
- Safe for educational and research use

---

## 8. Reproducibility
To fully reproduce the model and all artifacts:


This executes:

1. Preprocessing
2. Hyperparameter tuning
3. Final training
4. Inference
5. Error analysis
6. Visualization

All outputs are stored in the `outputs/` directory.

---

## 9. Artifacts
- `models/best_model.pt` — trained MLP model
- `configs/best_config.json` — best hyperparameters
- `data/processed/` — preprocessed datasets
- `outputs/predictions.csv` — inference results
- `outputs/error_analysis.json` — error statistics
- `outputs/plots/` — confusion matrix, F1 charts, hardest classes, error heatmaps

---

## 10. Known Failure Modes
- Confusion between visually or structurally similar letters
- Reduced performance on noisy or shifted feature distributions
- No robustness to adversarial or corrupted inputs
- No built‑in uncertainty estimation

---

## 11. Author
**Dominik**
Machine Learning Engineer (Portfolio Project)
