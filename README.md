# MLP Training Pipeline — OpenML Letter Recognition

A learning project: an end-to-end training pipeline for a **PyTorch MLP** that classifies letters **A–Z** from 16 numeric features.
The goal was to learn the full ML workflow — EDA, data splitting, hyperparameter tuning with cross-validation, training, evaluation and error analysis — and to compare the result with simple baselines.

A separate demo API that serves this model (self-hosted, not production): [api-mlp-project](https://github.com/Doomiiniik/api-mlp-project)

---

## Dataset

- **OpenML Letter Recognition** (dataset ID 6): 20,000 samples, 26 classes (A–Z)
- Images of **printed** letters (20 fonts, randomly distorted), already converted to **16 integer features (0–15)** describing shape and pixel statistics. The model does not see images.
- After removing **1,332 duplicate rows**: 18,668 samples

## Results (test set, 3,734 samples)

| Metric | Value |
|---|---|
| Accuracy | **95.8%** (157 errors) |
| Macro F1 | 0.958 |
| Weighted F1 | 0.958 |

### Baselines (same train/test split, default settings, no tuning)

| Model | Test accuracy | Fit time |
|---|---|---|
| Logistic Regression | 75.6% | ~3 s |
| Random Forest | 95.7% | ~2 s |
| HistGradientBoosting | 95.7% | ~9 s |
| **MLP (this project)** | **95.8%** | ~22 min tuning + ~1 min training |

**Takeaway:** on this tabular dataset a default Random Forest matches the tuned MLP. For a real use case I would pick a tree-based model; the MLP was chosen to learn PyTorch and the training process.

Run the baselines: `python -m scripts.run_baseline`

---

## Pipeline

1. **Preprocessing** (`scripts/preprocess_data.py`)
   - download from OpenML (or load cached CSV), remove duplicates
   - encode labels with `LabelEncoder` (saved to `data/processed/label_encoder.pkl`)
   - stratified split: **72% train / 8% val / 20% test** (13,440 / 1,494 / 3,734)
   - `StandardScaler` fitted **on train only**, applied to val and test (saved to `models/scaler.pkl`)
2. **Tuning** (`scripts/tune_model.py`)
   - Optuna, TPE sampler (seeded), **20 trials**
   - each trial: **5-fold stratified CV on the train set**, 10 epochs per fold; objective = mean of the best validation loss per fold
   - search space: hidden layers from `[64]`, `[128]`, `[256]`, `[64, 32]`, `[128, 64]`, `[256, 128]`, `[128, 128, 64]`; dropout 0–0.5; learning rate 1e-4–1e-2 (log scale); batch size {32, 64, 128, 256}; gradient clipping 0.5–5.0
   - best config (`configs/best_config.json`): `[256, 128]`, dropout ≈ 0.10, lr ≈ 0.0048, batch 64, grad clip ≈ 3.85
3. **Final training** (`scripts/train_model.py`)
   - Adam, CrossEntropyLoss, gradient clipping
   - trained on train, **early stopping on val** (patience 7, max 100 epochs); the best checkpoint is saved
   - latest run: best epoch 30, stopped at epoch 37
4. **Inference** (`scripts/run_inference.py`) → `outputs/predictions.csv`
5. **Error analysis** (`scripts/run_error_analysis.py`) → per-class F1, hardest classes, top confusion pairs (`outputs/error_analysis.json`)
6. **Visualizations** (`scripts/run_visualizations.py`) → confusion matrix and F1 plots in `outputs/plots/`

EDA is in `notebooks/01_eda.ipynb` (missing values, duplicates, distributions, skewness/kurtosis, correlations).

## Error analysis

- Hardest classes (lowest F1): **B (0.923)**, **H (0.925)**, R (0.938)
- Most frequent confusions (true → predicted, counted over all errors): **S → B (7)**, **H → N (6)**, **K → R (6)**, F → P (5), J → I (5)

---

## How to run

Run all commands from the repository root.

```bash
pip install -r requirements.txt          # pipeline
pip install -r requirements-dev.txt      # + notebook tools

python run_pipeline.py --all             # full pipeline (tuning takes ~20+ min)
```

Or single steps: `--preprocess`, `--tune`, `--train`, `--infer`, `--error`, `--viz`.

Trained artifacts (`models/best_model.pt`, `models/scaler.pkl`, `data/processed/label_encoder.pkl`) are included, so `--infer --error --viz` work right after cloning.

## Project structure

```
mlp-openml-project/
├── configs/        # best hyperparameters, Optuna trials
├── data/processed/ # train/val/test CSVs (scaled), label encoder
├── models/         # trained model, scaler
├── notebooks/      # EDA
├── outputs/        # predictions, error analysis, plots, baseline results
├── reports/        # model card
├── scripts/        # pipeline steps (entry points)
├── src/            # reusable code: data, model, training, tuning
└── run_pipeline.py # runs the steps in order
```

---

## Limitations and what I would do differently

- **MLP ≈ Random Forest** here; trees are simpler, faster and need no scaling.
- **Small tuning budget:** 20 trials, the first ~10 are random (TPE startup); single-layer architectures were never sampled. More trials with pruning would explore the space better.
- **Epoch mismatch:** tuning evaluates 10 epochs per fold, while the final model trains ~30 epochs. Tuning the number of epochs (or early stopping inside CV) would be more consistent.
- **Noisy validation:** the val set is small (~57 samples per class), so val loss fluctuates between epochs.
- Some duplicated logic remains (e.g. inference code in `src/inference.py` is not used by the scripts); no unit tests or CI yet.
- The model works only on these 16 precomputed features — it is not an OCR model and does not work on raw images or handwriting.

## Author

Dominik Wiśniowski — Medical Informatics student, AGH University of Krakow
