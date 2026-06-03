# Customer Churn Prediction Project

This project predicts customer churn for a telecom company using a Gradient Boosting Classifier. It includes a Jupyter Notebook for exploratory data analysis (EDA) & modeling, and a Streamlit web application for interactive predictions.

## Project Structure

- `chrun.ipynb`: Jupyter Notebook containing EDA, feature engineering, and model training.
- `app.py`: Streamlit application for predicting customer churn.
- `churn_model.pkl`: Trained Gradient Boosting Classifier.
- `Churn_dataset.csv`: Original Telco Customer Churn dataset.
- `Churn_engineered_full.csv`: Feature-engineered dataset (all columns).
- `Churn_engineered_final.csv`: Feature-engineered dataset (only columns used for model training).

## How to Run Streamlit App

1. Install dependencies:
   ```bash
   pip install streamlit pandas numpy joblib scikit-learn
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
