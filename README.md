# Benchmarking Supervised Learning Algorithms

> **📄 [Read the full report (PDF)](report/main.pdf)**

Comparative analysis of four supervised learning algorithm families on two classification tasks, evaluating Decision Trees, k-Nearest Neighbors, Support Vector Machines, and Neural Networks (scikit-learn and PyTorch).

## Datasets

| Dataset | Samples | Features | Task | Metric |
|---------|---------|----------|------|--------|
| **Adult Income** | 45,222 | 14 (→104 after encoding) | Binary classification | F1 Score |
| **Wine Quality** | 6,497 | 13 | 8-class classification | Macro-F1 |

## Results Summary

| Algorithm | Adult F1 | Wine Macro-F1 |
|-----------|----------|---------------|
| Decision Tree | 0.670 | 0.432 |
| k-Nearest Neighbors | 0.642 | 0.455 |
| SVM (best kernel) | 0.654 | 0.467 |
| Neural Network (sklearn) | 0.681 | 0.285 |
| Neural Network (PyTorch) | 0.673 | 0.309 |

## Key Findings

- **Neural networks achieve the highest F1 on Adult** (0.681), but the margin over simpler models is small — confirming that the high-dimensional one-hot encoded space is approximately linearly separable.
- **SVM (RBF) wins on Wine** (Macro-F1 = 0.467), where non-linear decision boundaries are critical for separating overlapping quality classes.
- **kNN surprises on Wine** (0.455) due to the low-dimensional continuous feature space, but fails on Adult due to the curse of dimensionality.
- **SGD-only neural networks struggle on small, imbalanced data** (Wine), demonstrating the importance of adaptive optimizers for practical deep learning.

## Project Structure

```
├── report/
│   ├── main.tex          # Full LaTeX report
│   ├── main.pdf          # Compiled report
│   └── figures/          # All learning curves, confusion matrices, etc.
├── notebooks/
│   └── analysis.ipynb    # Complete analysis notebook
├── data/
│   ├── adult.csv
│   └── wine.csv
└── requirements.txt
```

## Tech Stack

Python, PyTorch, scikit-learn, Optuna, Pandas, NumPy, Matplotlib, Seaborn

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```
