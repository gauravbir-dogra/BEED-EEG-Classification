import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="BEED EEG Classification",
    layout="wide"
)

st.title("BEED EEG Epilepsy Classification")

st.write(
    "Upload the test dataset and evaluate the saved "
    "machine learning classification models."
)


# --------------------------------------------------
# Load Saved Models
# --------------------------------------------------

models = {
    "Logistic Regression":
        joblib.load("model/logistic_regression.joblib"),

    "Decision Tree":
        joblib.load("model/decision_tree.joblib"),

    "K-Nearest Neighbors (KNN)":
        joblib.load("model/knn.joblib"),

    "Gaussian Naive Bayes":
        joblib.load("model/gaussian_nb.joblib"),

    "Support Vector Machine (SVM)":
        joblib.load("model/svm.joblib")
}

scaler = joblib.load("model/scaler.joblib")


# Models that require scaled features
scaled_models = [
    "Logistic Regression",
    "K-Nearest Neighbors (KNN)",
    "Support Vector Machine (SVM)"
]


# --------------------------------------------------
# Dataset Upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test data in CSV format",
    type=["csv"]
)


if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully.")

    st.dataframe(
        test_data.head(),
        use_container_width=True
    )


    # --------------------------------------------------
    # Validate Required Columns
    # --------------------------------------------------

    features = [f"X{i}" for i in range(1, 17)]

    required_columns = features + ["y"]

    missing_columns = [
        column
        for column in required_columns
        if column not in test_data.columns
    ]

    if missing_columns:

        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

        st.stop()


    # --------------------------------------------------
    # Separate Features and Target
    # --------------------------------------------------

    X_test_app = test_data[features]
    y_test_app = test_data["y"]

    X_test_scaled_app = scaler.transform(X_test_app)


    # --------------------------------------------------
    # Evaluation Function
    # --------------------------------------------------

    def evaluate_model(model_name):

        model = models[model_name]

        if model_name in scaled_models:
            X_eval = X_test_scaled_app
        else:
            X_eval = X_test_app

        y_pred = model.predict(X_eval)

        y_prob = model.predict_proba(X_eval)

        accuracy = accuracy_score(
            y_test_app,
            y_pred
        )

        auc = roc_auc_score(
            y_test_app,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

        precision = precision_score(
            y_test_app,
            y_pred,
            average="macro"
        )

        recall = recall_score(
            y_test_app,
            y_pred,
            average="macro"
        )

        f1 = f1_score(
            y_test_app,
            y_pred,
            average="macro"
        )

        mcc = matthews_corrcoef(
            y_test_app,
            y_pred
        )

        metrics = {
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc
        }

        return metrics, y_pred


    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------

    st.header("2. Model Selection")

    selected_model = st.selectbox(
        "Select Classification Model",
        list(models.keys())
    )


    # --------------------------------------------------
    # Selected Model Evaluation
    # --------------------------------------------------

    metrics, predictions = evaluate_model(
        selected_model
    )


    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------

    st.header("3. Evaluation Metrics")

    st.subheader(selected_model)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.4f}"
    )

    col2.metric(
        "AUC",
        f"{metrics['AUC']:.4f}"
    )

    col3.metric(
        "Precision",
        f"{metrics['Precision']:.4f}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{metrics['Recall']:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.4f}"
    )

    col6.metric(
        "MCC",
        f"{metrics['MCC']:.4f}"
    )


    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    st.header("4. Confusion Matrix")

    cm = confusion_matrix(
        y_test_app,
        predictions,
        labels=[0, 1, 2, 3]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1",
            "Actual 2",
            "Actual 3"
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
            "Predicted 2",
            "Predicted 3"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


    # --------------------------------------------------
    # Results for All Models
    # --------------------------------------------------

    st.header("5. Results of All Models")

    comparison_results = []

    for model_name in models.keys():

        model_metrics, _ = evaluate_model(
            model_name
        )

        comparison_results.append({
            "Model": model_name,
            "Accuracy": model_metrics["Accuracy"],
            "AUC": model_metrics["AUC"],
            "Precision": model_metrics["Precision"],
            "Recall": model_metrics["Recall"],
            "F1 Score": model_metrics["F1 Score"],
            "MCC": model_metrics["MCC"]
        })

    comparison_df = pd.DataFrame(
        comparison_results
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "Upload test_data.csv to evaluate the saved models."
    )
