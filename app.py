import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="BEED EEG Classification",
    layout="wide"
)


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

log_model = joblib.load("model/logistic_regression.joblib")
dt_model = joblib.load("model/decision_tree.joblib")
knn_model = joblib.load("model/knn.joblib")
nb_model = joblib.load("model/gaussian_nb.joblib")
svm_model = joblib.load("model/svm.joblib")

# Fitted preprocessing scaler
scaler = joblib.load("model/scaler.joblib")


# --------------------------------------------------
# Class labels
# --------------------------------------------------

class_names = {
    0: "Healthy Subject",
    1: "Generalized Seizure",
    2: "Focal Seizure",
    3: "Seizure Event"
}


# --------------------------------------------------
# Application heading
# --------------------------------------------------

st.title("BEED EEG Epilepsy Classification")

st.write(
    "Enter the 16 EEG feature values and select a classification "
    "model to predict the EEG category."
)


# --------------------------------------------------
# Model selection
# --------------------------------------------------

selected_model = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Gaussian Naive Bayes",
        "SVM"
    ]
)


# --------------------------------------------------
# EEG feature inputs
# --------------------------------------------------

st.subheader("EEG Feature Inputs")

feature_values = {}

columns = st.columns(4)

for i in range(16):

    feature_name = f"X{i + 1}"

    with columns[i % 4]:

        feature_values[feature_name] = st.number_input(
            feature_name,
            value=0,
            step=1,
            format="%d"
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict EEG Class"):

    input_data = pd.DataFrame(
        [feature_values],
        columns=[f"X{i}" for i in range(1, 17)]
    )

    if selected_model == "Logistic Regression":

        input_processed = scaler.transform(input_data)

        prediction = log_model.predict(
            input_processed
        )[0]

    elif selected_model == "Decision Tree":

        prediction = dt_model.predict(
            input_data
        )[0]

    elif selected_model == "KNN":

        input_processed = scaler.transform(input_data)

        prediction = knn_model.predict(
            input_processed
        )[0]

    elif selected_model == "Gaussian Naive Bayes":

        prediction = nb_model.predict(
            input_data
        )[0]

    elif selected_model == "SVM":

        input_processed = scaler.transform(input_data)

        prediction = svm_model.predict(
            input_processed
        )[0]

    st.success(
        f"Predicted Class: {prediction} - "
        f"{class_names[prediction]}"
    )
