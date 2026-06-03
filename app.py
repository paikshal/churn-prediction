import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ============================================================
# Load Saved Model
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# ============================================================
# App Header
# ============================================================
st.title("📡 Customer Churn Prediction App")
st.markdown("### Predict whether a customer will leave the company or stay.")
st.divider()

# ============================================================
# Sidebar - User Input Fields
# ============================================================
st.sidebar.header("🧾 Enter Customer Information")

# --- Contract & Tenure ---
contract = st.sidebar.selectbox(
    "Contract Type",
    options=["Month-to-month", "One year", "Two year"],
    help="How long is the customer's contract?"
)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract_encoded = contract_map[contract]

tenure = st.sidebar.slider(
    "Tenure (Months)",
    min_value=0, max_value=72, value=12,
    help="How many months has the customer been with the company?"
)

# --- Charges ---
monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=18.0, max_value=120.0, value=65.0, step=0.5,
    help="Monthly bill of the customer"
)

# --- Internet Service ---
internet_service = st.sidebar.selectbox(
    "Internet Service",
    options=["DSL", "Fiber optic", "No"],
    help="Which internet service does the customer use?"
)
internet_fiber = 1 if internet_service == "Fiber optic" else 0
internet_no = 1 if internet_service == "No" else 0

# --- Online Security ---
online_security = st.sidebar.selectbox(
    "Online Security",
    options=["Yes", "No", "No internet service"],
    help="Does the customer have Online Security?"
)
online_security_encoded = 1 if online_security == "Yes" else 0

# --- Tech Support ---
tech_support = st.sidebar.selectbox(
    "Tech Support",
    options=["Yes", "No", "No internet service"],
    help="Does the customer use Tech Support?"
)
tech_support_encoded = 1 if tech_support == "Yes" else 0

# --- Billing ---
paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    options=["Yes", "No"],
    help="Is the customer's billing paperless?"
)
paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0

# --- Payment Method ---
payment_method = st.sidebar.selectbox(
    "Payment Method",
    options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    help="How does the customer make payments?"
)
is_autometic_payment = 1 if "automatic" in payment_method.lower() else 0

# --- Demographics ---
senior_citizen = st.sidebar.selectbox(
    "Senior Citizen?",
    options=["No", "Yes"],
    help="Is the customer a senior citizen?"
)
senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0

# Note: Partner and Dependents were not in model training, so they were removed

# ============================================================
# Prepare Input Data for Model
# ============================================================
input_data = pd.DataFrame({
    "Contract": [contract_encoded],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "InternetService_No": [internet_no],
    "InternetService_Fiber optic": [internet_fiber],
    "OnlineSecurity": [online_security_encoded],
    "TechSupport": [tech_support_encoded],
    "PaperlessBilling": [paperless_billing_encoded],
    "SeniorCitizen": [senior_citizen_encoded],
    "is_Autometic_payment": [is_autometic_payment]  # Typo in training column name
})

# ============================================================
# Main Page - Display Input Summary & Prediction
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Customer Summary")
    st.table(input_data.T.rename(columns={0: "Value"}))

with col2:
    st.subheader("🔮 Prediction Result")

    if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
        # Model prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        churn_prob = prediction_proba[1] * 100
        no_churn_prob = prediction_proba[0] * 100

        if prediction == 1:
            st.error(f"⚠️ **Customer will CHURN!**")
            st.metric(
                label="Churn Probability",
                value=f"{churn_prob:.1f}%",
                delta="High Risk",
                delta_color="inverse"
            )
        else:
            st.success(f"✅ **Customer will NOT Churn!**")
            st.metric(
                label="Churn Probability",
                value=f"{churn_prob:.1f}%",
                delta="Low Risk",
                delta_color="normal"
            )

        # Probability Bar
        st.write("**Probability Breakdown:**")
        st.progress(int(churn_prob), text=f"Churn Risk: {churn_prob:.1f}%")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"✅ No Churn: **{no_churn_prob:.1f}%**")
        with col_b:
            st.warning(f"⚠️ Churn: **{churn_prob:.1f}%**")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("Model: Gradient Boosting Classifier | Dataset: Telco Customer Churn")
