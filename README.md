# Gallstone Prediction Using Machine Learning and Explainable AI

A machine learning project that predicts **gallstone status** from patient clinical measurements, body composition metrics, and laboratory results. The project compares four classification algorithms and uses Explainable AI (XAI) to investigate which features influence model predictions.

> **Disclaimer:** This is an educational machine learning project, not a clinically validated diagnostic system. Its predictions must not be used for medical decisions.

## Project overview

The goal was to explore whether routinely collected patient information could be used to classify gallstone status (`0`: no gallstones; `1`: gallstones). I built an end-to-end workflow covering data checks, feature selection, model training, hyperparameter tuning, test-set evaluation, and model explanation.

## Dataset

- **319 records** and **39 columns** in the original dataset, including the target column.
- Inputs include age, sex, existing health conditions, body composition measurements, and blood test results.
- Example features: C-reactive protein (CRP), vitamin D, extracellular fluid/total body water (ECF/TBW), cholesterol measurements, and visceral fat area.
- Target: **Gallstone Status** (binary classification).


## Tools and libraries

**Python**, **pandas**, **NumPy**, **scikit-learn**, **SciPy**, **LIME**, and **Matplotlib**.

## Methodology

### 1. Data preparation

- Checked for duplicate records and missing values; none were found in the analyzed dataset.
- Flagged potential outliers using absolute Z-scores greater than 3; kept them rather than automatically removing potentially meaningful clinical measurements.
- Removed `Height`, `Weight`, `Extracellular Water (ECW)`, and `Intracellular Water (ICW)` to reduce redundant inputs.
- Split the data into **80% training** and **20% testing**, stratified by gallstone status (`random_state=42`).
- Used `StandardScaler` inside scikit-learn pipelines, fitting preprocessing within the model training and validation workflow.

### 2. Model training and tuning

Compared four supervised classification models:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. K-Nearest Neighbors (KNN)

Used **GridSearchCV with 5-fold cross-validation** to tune model hyperparameters and select the best settings based on validation accuracy.

### 3. Model evaluation

Evaluated each selected model on the held-out test set using accuracy, precision, recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | **81.25%** | 0.794 | 0.844 | 0.818 |
| Random Forest | **81.25%** | 0.794 | 0.844 | 0.818 |
| SVM | 79.69% | 0.788 | 0.813 | 0.800 |
| KNN | 68.75% | 0.833 | 0.469 | 0.600 |

*These are results reported from the project's test evaluation, not estimates of performance in real-world clinical use.*

Logistic Regression and Random Forest achieved the highest test accuracy. KNN had lower recall, meaning it missed more actual positive cases. The Random Forest's training accuracy was higher than its cross-validation accuracy, suggesting a risk of overfitting.

For the Random Forest, the reported test confusion matrix was:

| | Predicted: No Gallstones | Predicted: Gallstones |
|---|---:|---:|
| **Actual: No Gallstones** | 25 | 7 |
| **Actual: Gallstones** | 5 | 27 |

### 4. Explainable AI

**Global feature importance — Random Forest**

I extracted the Random Forest model's built-in feature importance scores to understand which variables were most useful for its decision-tree splits. The highest-ranked features included:

- C-reactive protein (CRP)
- Vitamin D
- Extracellular Fluid / Total Body Water (ECF/TBW)
- Aspartate Aminotransferase (AST)
- Visceral Muscle Area (VMA)

These are *model importance scores*, not proof that the features cause gallstones.

**Local explanations — LIME**

I used **LIME (Local Interpretable Model-agnostic Explanations)** to inspect a single Random Forest prediction. In the example visualization, the model assigned:

- **76%** probability to *No Gallstones*
- **24%** probability to *Gallstones*

LIME illustrated which input features pushed this particular prediction toward either class. The explanation applies to that observation and is not a general medical rule.

## Limitations and future work

- The dataset is small (319 records), so the results may not generalize to other patient populations.
- Validation and external clinical testing would be needed before considering any real-world healthcare application.
- Random Forest showed signs of possible overfitting based on training versus cross-validation performance.
- Future work could explore larger and more diverse datasets, calibration of predicted probabilities, and independent external validation.

## What I learned

This project helped me practice the full supervised learning workflow: preparing clinical tabular data, preventing data leakage through preprocessing pipelines, comparing classification algorithms, interpreting evaluation metrics, and explaining individual model predictions using LIME.

