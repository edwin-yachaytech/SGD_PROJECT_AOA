# Stochastic Gradient Descent Final Project

This repository contains a compact implementation of the final project:
**Analysis of Stochastic Gradient Descent for Logistic Regression on Medical Data**.

The project studies SGD as an algorithm, focusing on:

- formal SGD update rule;
- logistic regression for binary classification;
- comparison with Batch Gradient Descent and Mini-Batch SGD;
- operation-level complexity analysis;
- empirical evaluation on the Breast Cancer Wisconsin Diagnostic data set.

## Files

- `run_sgd_final.py`: complete implementation and experiments in one Python file.
- `wdbc.data`: Breast Cancer Wisconsin Diagnostic data set from UCI.
- `requirements.txt`: Python dependencies.

## Dataset

The data set is the Breast Cancer Wisconsin Diagnostic (WDBC) data set.

- Source: UCI Machine Learning Repository
- Instances: 569
- Features: 30 continuous features
- Classes: malignant and benign
- Encoding in the code: malignant = 1, benign = 0

Original source:
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python run_sgd_final.py
```

The script trains:

- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch SGD

It prints classification metrics, runtime, update counts, and theoretical
dominant operation costs.

## Main Complexity Result

For one dense sample with `d` features, one SGD update computes:

- dot product `w^T x_i`: `O(d)`;
- sigmoid and scalar error: `O(1)`;
- gradient vector `(p_i - y_i)x_i`: `O(d)`;
- weight update: `O(d)`.

Therefore:

```text
T_SGD_update(d) = O(d)
```

Batch Gradient Descent uses all `n` samples before one update:

```text
T_Batch_update(n,d) = O(nd)
```

Mini-Batch SGD with batch size `b` costs:

```text
T_MiniBatch_update(b,d) = O(bd)
```

## Main Result

On the selected 80/20 train-test split, the three optimizers reached
approximately 96.49% test accuracy.
