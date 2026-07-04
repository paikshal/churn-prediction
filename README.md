# 📡 Customer Churn Prediction System

> **An end-to-end Machine Learning web application** that predicts telecom customer churn using a Gradient Boosting Classifier with an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 

**Customer Churn Prediction (Telecom):** Built an end-to-end binary classification system on a class-imbalanced dataset (73.5% No Churn, 26.5% Churn) using a **Gradient Boosting Classifier** trained on 7,043 customer records across 21 features. Performed feature engineering — created `is_Automatic_Payment` binary flag, bucketed `tenure` into 4 lifecycle groups (0–12, 13–24, 25–48, 48+ months), and applied one-hot encoding — reducing the feature space from 21 to 10 high-impact predictors. Deployed the model as a **Streamlit web app** with real-time churn probability scoring and a 7-view interactive analytics dashboard supporting segment-level filtering, correlation heatmaps, KDE plots, and tenure group analysis.

---

## 🚀 Project Overview

Customer churn — when customers leave a company — is one of the most critical business problems in the telecom industry. This project presents an **ML-powered web application** that:

- **Predicts in real-time** whether a customer is likely to churn or stay
- Provides a **churn probability score** (e.g., "78.5% chance of churn")
- Offers an **interactive analytics dashboard** to explore patterns in the dataset

**Business Impact:** By identifying and retaining just 1,000 at-risk customers, a telecom company could save approximately **$50,000–$150,000 per month** in lost revenue (based on an average of $65/month per customer).

---

## 🎯 Key Features

| Feature                                     | Description                                                                |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| 🔮**Real-time Churn Prediction**      | Enter 10 customer attributes and get an instant churn prediction           |
| 📊**Interactive Analytics Dashboard** | 7 EDA visualizations with dynamic filtering options                        |
| 🔍**Segment-Level Analysis**          | Dataset automatically filters based on the sidebar prediction profile      |
| 📈**Probability Score**               | Returns not just Yes/No, but an actual churn probability percentage        |
| 🎛️**Customer Profiling**            | Analysis across contract type, tenure, charges, internet service, and more |

---

## 🛠️ Tech Stack

| Layer                         | Technology                                  |
| ----------------------------- | ------------------------------------------- |
| **Language**            | Python 3.10+                                |
| **ML Model**            | Gradient Boosting Classifier (scikit-learn) |
| **Web Framework**       | Streamlit                                   |
| **Data Processing**     | Pandas, NumPy                               |
| **Visualization**       | Matplotlib, Seaborn                         |
| **Model Serialization** | Joblib (.pkl)                               |
| **Dataset**             | IBM Telco Customer Churn (~7,000 records)   |

---

## 📂 Project Structure

```
churn-prediction/
│
├── app.py                      # Streamlit web application (main UI)
├── chrun.ipynb                 # Jupyter Notebook — EDA, Feature Engineering & Model Training
├── churn_model.pkl             # Trained Gradient Boosting Classifier (saved model)
│
├── Churn_dataset.csv           # Original raw dataset (~7,000 customers)
├── Churn_engineered_full.csv   # Feature-engineered dataset (all columns)
├── Churn_engineered_final.csv  # Final dataset used for model training
│
└── requirements.txt            # All project dependencies
```

---

## 🧠 ML Pipeline & Methodology

### 1. Data Collection & EDA

- Used the IBM Telco Customer Churn dataset — **7,043 customer records**, 21 features
- Analyzed churn distribution (**~26.5% churn rate** — class imbalance identified)
- Explored numeric features: `tenure`, `MonthlyCharges`, `TotalCharges`
- Examined categorical features: `Contract`, `InternetService`, `PaymentMethod`, and others

### 2. Feature Engineering

Key transformations derived from EDA:

- Created **Tenure Group** bins (0–12, 13–24, 25–48, 48+ months)
- Engineered **is_Automatic_payment** binary flag (automatic vs. manual payment)
- Encoded categorical variables using Label Encoding and One-Hot Encoding
- Removed highly correlated and low-importance features

### 3. Model Selection & Training

- Trained and compared multiple classifiers
- Selected **Gradient Boosting Classifier** as the final model based on performance
- Final model uses 10 high-impact features:
  - `Contract`, `tenure`, `MonthlyCharges`, `InternetService_No`, `InternetService_Fiber optic`
  - `OnlineSecurity`, `TechSupport`, `PaperlessBilling`, `SeniorCitizen`, `is_Automatic_payment`

### 4. Model Deployment

- Model serialized using `joblib` → `churn_model.pkl`
- Loaded in Streamlit via `@st.cache_resource` for optimized, production-level performance

---

## 📊 Analytics Dashboard — Visualizations

The dashboard provides **7 dynamic analysis views**:

1. **Overall Churn Distribution** — Count plot + Pie chart showing the churn ratio
2. **Numeric Features vs Churn** — Boxplots & KDE density plots for tenure and charges
3. **Tenure Group Analysis** — Churn rate broken down by customer tenure range
4. **Categorical Features Churn Rate** — Bar charts for Contract, Internet Service, and Payment Method
5. **Service Features Churn Rate** — Churn rate when a specific service is active (e.g., Tech Support, Streaming TV)
6. **Correlation Matrix & Heatmap** — Feature-level correlation with churn
7. **Pairplot** — Multivariate relationship between key numeric features

---

## 🏃 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉


---
