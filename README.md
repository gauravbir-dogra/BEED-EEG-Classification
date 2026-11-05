# BEED EEG Epilepsy Classification

## a. Problem Statement

Epilepsy is a neurological disorder associated with abnormal electrical activity in the brain, which may result in recurrent seizures. Electroencephalography (EEG) records electrical brain activity and can provide useful information for identifying seizure-related patterns.

The objective of this project is to develop, evaluate, and compare multiple machine learning classification models for classifying EEG recordings from the Bangalore EEG Epilepsy Dataset (BEED).

The following five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Model performance was evaluated using six metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Hyperparameter tuning was performed using cross-validation where appropriate to improve model generalization.

---

## b. Dataset Description

The Bangalore EEG Epilepsy Dataset (BEED) is a multiclass EEG dataset designed for epilepsy and seizure classification.

### Dataset Characteristics

- **Number of observations:** 8,000
- **Number of predictor variables:** 16
- **Predictors:** `X1` to `X16`
- **Target variable:** `y`
- **Number of classes:** 4
- **Observations per class:** 2,000

### Target Classes

| Class | EEG Category |
|---:|---|
| 0 | Healthy Subjects |
| 1 | Generalized Seizures |
| 2 | Focal Seizures |
| 3 | Seizure Events |

### Data Preparation

The dataset was divided into training and testing datasets using an **80:20 stratified split**.

This resulted in:

- **Training set:** 6,400 observations
- **Test set:** 1,600 observations
- **Training observations per class:** 1,600
- **Test observations per class:** 400

Stratification ensured that the original class distribution was maintained in both training and test datasets.

Feature standardization using `StandardScaler` was applied to algorithms that are sensitive to feature magnitude:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

The original unscaled features were used for:

- Decision Tree
- Gaussian Naive Bayes

### Dataset Source

[Bangalore EEG Epilepsy Dataset (BEED) - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset)

---

## c. GitHub Repository Link

**GitHub Repository:**  
https://github.com/gauravbir-dogra/BEED-EEG-Classification

---

## d. Models Used

Five machine learning classification algorithms were developed and compared.

Hyperparameter tuning using cross-validation was performed where appropriate. The final model configurations were evaluated using the same held-out test dataset.

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4944 | 0.6472 | 0.5021 | 0.4944 | 0.4869 | 0.3309 |
| Decision Tree | 0.8894 | 0.9262 | 0.8900 | 0.8894 | 0.8897 | 0.8525 |
| **K-Nearest Neighbors (KNN)** | **0.9888** | 0.9925 | **0.9889** | **0.9888** | **0.9888** | **0.9850** |
| Gaussian Naive Bayes | 0.6975 | 0.9062 | 0.6948 | 0.6975 | 0.6925 | 0.5990 |
| Support Vector Machine (SVM) | 0.9881 | **0.9981** | 0.9882 | 0.9881 | 0.9882 | 0.9842 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression produced the lowest overall predictive performance. Regularization tuning improved test accuracy from approximately **45.44% to 49.44%**, but performance remained considerably below the nonlinear classifiers. This suggests that linear decision boundaries were insufficient to capture the class structure present in the EEG features. |
| **Decision Tree** | Decision Tree captured nonlinear relationships substantially better than Logistic Regression and achieved **88.94% accuracy** with an F1 score of **0.8897**. The initial unrestricted tree achieved 100% training accuracy, indicating potential overfitting. Complexity-related hyperparameters were therefore evaluated to improve model generalization. |
| **K-Nearest Neighbors (KNN)** | KNN achieved the strongest overall performance. Hyperparameter tuning selected **1 nearest neighbor (`k=1`) with Manhattan distance (`p=1`)**. The final model achieved **98.88% accuracy**, **98.89% precision**, **98.88% recall**, **98.88% F1**, and the highest MCC of **0.9850**. |
| **Gaussian Naive Bayes** | Gaussian Naive Bayes achieved **69.75% accuracy** and an AUC of **0.9062**. The `var_smoothing` parameter was tuned using cross-validation, but tuning did not improve final test accuracy. The comparatively lower performance may reflect limitations of the conditional-independence assumption for the EEG predictors. |
| **Support Vector Machine (SVM)** | The tuned SVM achieved **98.81% accuracy**, **98.82% F1**, and an MCC of **0.9842**. SVM achieved the highest AUC of all five classifiers at **0.9981**, indicating excellent discrimination among the four EEG classes. Its overall predictive performance was extremely close to KNN. |

---

## Overall Winner for the Dataset

### K-Nearest Neighbors (KNN)

KNN was selected as the **overall best-performing classifier** because it achieved the highest values for five of the six evaluation metrics.

| Metric | KNN Result |
|---|---:|
| Accuracy | **0.9888** |
| AUC | 0.9925 |
| Precision | **0.9889** |
| Recall | **0.9888** |
| F1 Score | **0.9888** |
| MCC | **0.9850** |

The tuned KNN model achieved approximately **98.88% test accuracy**.

Although SVM achieved the highest AUC of **0.9981**, KNN achieved slightly higher Accuracy, Precision, Recall, F1 Score, and MCC.

Therefore, **KNN was selected as the overall winner for this dataset**.

---

## Repository Structure

```text
BEED-EEG-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── BEED_EEG_Classification.ipynb
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_nb.joblib
    ├── svm.joblib
    └── scaler.joblib
```

---

## Streamlit Application

The trained machine learning models are deployed using Streamlit Community Cloud.

The application is designed to evaluate the saved classification models on an uploaded test dataset without retraining any model.

### Application Features

The Streamlit application allows users to:

1. Upload the test dataset in CSV format.
2. Validate that the uploaded dataset contains all required predictors (`X1` to `X16`) and the target variable (`y`).
3. Select one of the five trained classification models from a dropdown:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
   - Support Vector Machine (SVM)
4. Evaluate the selected model using:
   - Accuracy
   - AUC
   - Precision
   - Recall
   - F1 Score
   - Matthews Correlation Coefficient (MCC)
5. Display the confusion matrix for the selected model.
6. View the evaluation results of all five models on the same uploaded test dataset.

The application loads previously trained and saved model artifacts. The uploaded test dataset is used only for model evaluation and is **not used to retrain or tune the models**.

### Test Dataset Format

The uploaded CSV file must contain the following columns:

```text
X1, X2, X3, X4, X5, X6, X7, X8,
X9, X10, X11, X12, X13, X14, X15, X16, y
```

The provided `test_data.csv` contains the held-out test observations used for evaluating the models.

### Preprocessing During Evaluation

The saved `StandardScaler` is applied to the uploaded feature data when evaluating:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

The original unscaled feature values are used for:

- Decision Tree
- Gaussian Naive Bayes

This ensures that each saved model receives data using the same preprocessing approach that was used during model development.


### Live Application

https://beed-eeg-classification-rbhtqzuveiss2fpvg2l8fu.streamlit.app

---
