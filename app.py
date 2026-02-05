import streamlit as st

st.title("Wine Quality Multiclass Classification")
st.write("Streamlit app setup successful!")


import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Wine Quality Classification", layout="centered")

st.title("🍷 Wine Quality Classification App")

st.write(
    "Upload a **CSV test dataset**, select a **model**, and view "
    "evaluation metrics and confusion matrix."
)

# --------------------------------------------------
# (a) DATASET UPLOAD (CSV)
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV only)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    df.columns = df.columns.str.strip()
 
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Split features and target
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # (b) MODEL SELECTION DROPDOWN
    # --------------------------------------------------
    model_name = st.selectbox(
        "Select Classification Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest"
        )
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # --------------------------------------------------
    # TRAIN & PREDICT (on uploaded test data)
    # --------------------------------------------------
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    # --------------------------------------------------
    # (c) DISPLAY EVALUATION METRICS
    # --------------------------------------------------
    st.subheader("📊 Evaluation Metrics")

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted")
    rec = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("Precision", f"{prec:.4f}")

    with col2:
        st.metric("Recall", f"{rec:.4f}")
        st.metric("F1-score", f"{f1:.4f}")

    # --------------------------------------------------
    # (d) CONFUSION MATRIX / CLASSIFICATION REPORT
    # --------------------------------------------------
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm)

    st.dataframe(cm_df)

    st.subheader("📋 Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.info("👆 Upload a CSV file to begin.")
