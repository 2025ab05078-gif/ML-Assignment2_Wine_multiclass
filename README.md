# Wine Quality Multiclass Classification

## a. Problem Statement
The objective of this project is to predict the quality of red wine based on
its physicochemical properties using multiple machine learning classification
models. This problem demonstrates the application of supervised learning
techniques and model comparison on a real-world multiclass dataset.

---

## b. Dataset Description
The Wine Quality (Red) dataset contains 1599 instances and 11 numerical input
features related to the chemical properties of wine such as acidity, sulphates,
alcohol content, and pH. The target variable is wine quality, which is an
integer score ranging from 3 to 8.  
Source: UCI Machine Learning Repository / Kaggle.

---

## c. Models Used and Performance Comparison

                     Accuracy  Precision  Recall  F1 Score     MCC     AUC
Logistic Regression    0.5906     0.5695  0.5906    0.5673  0.3250  0.7555
Decision Tree          0.6094     0.6121  0.6094    0.6095  0.3982  0.6991
KNN                    0.6094     0.5841  0.6094    0.5959  0.3733  0.7476
Naive Bayes            0.5625     0.5745  0.5625    0.5681  0.3299  0.7377
Random Forest          0.6750     0.6504  0.6750    0.6603  0.4768  0.8381
XGBoost                0.6531     0.6480  0.6531    0.6434  0.4453  0.8153

---

## d. Model Observations

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | Performed reasonably well but struggled to capture complex nonlinear relationships in the data. |
| Decision Tree | Captured nonlinear patterns but showed signs of overfitting. |
| KNN | Performance was sensitive to feature scaling and choice of k. |
| Naive Bayes | Fast and simple but assumptions of feature independence limited performance. |
| Random Forest (Ensemble) | Provided better generalization and improved accuracy due to ensemble learning. |
| XGBoost (Ensemble) | Achieved the best overall performance by effectively modeling complex patterns and handling class imbalance. |

---

## Deployment
The application is deployed using Streamlit Community Cloud and allows users to
upload test data, select models, view evaluation metrics, and analyze prediction
results interactively.
