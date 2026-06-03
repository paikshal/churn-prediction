import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
# Main Page Tabs (Prediction & Dashboard)
# ============================================================
tab1, tab2 = st.tabs(["🔮 Predict Churn", "📊 Analytics Dashboard"])

with tab1:
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

with tab2:
    st.header("📊 Customer Churn Analytics Dashboard")
    st.markdown("This dashboard displays insights and patterns from the Telco Customer Churn dataset.")

    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv("Churn_dataset.csv")

    try:
        df_original = load_data()

        # Filtering options
        st.subheader("🔍 Filter Options")
        
        # Sidebar/Prediction Profile sync toggle
        filter_mode = st.radio(
            "Select Filtering Mode:",
            options=["Show All Data", "Sync with Sidebar Prediction Profile", "Custom Manual Filters"],
            horizontal=True
        )

        df_data = df_original.copy()

        if filter_mode == "Sync with Sidebar Prediction Profile":
            st.info(f"Filtering dataset to match: Contract='{contract}', InternetService='{internet_service}', OnlineSecurity='{online_security}', TechSupport='{tech_support}', PaperlessBilling='{paperless_billing}', PaymentMethod='{payment_method}', SeniorCitizen='{senior_citizen}'")
            
            # Perform filtering on categorical fields
            df_data = df_data[
                (df_data['Contract'] == contract) &
                (df_data['InternetService'] == internet_service) &
                (df_data['OnlineSecurity'] == online_security) &
                (df_data['TechSupport'] == tech_support) &
                (df_data['PaperlessBilling'] == paperless_billing) &
                (df_data['SeniorCitizen'] == (1 if senior_citizen == "Yes" else 0)) &
                (df_data['PaymentMethod'] == payment_method)
            ]
            
            if len(df_data) == 0:
                st.warning("⚠️ No exact matches found for this profile in the dataset. Showing fallback results by filtering only Contract and Internet Service.")
                df_data = df_original[
                    (df_original['Contract'] == contract) &
                    (df_original['InternetService'] == internet_service)
                ]

        elif filter_mode == "Custom Manual Filters":
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                selected_contracts = st.multiselect(
                    "Contract Type",
                    options=df_original['Contract'].unique(),
                    default=df_original['Contract'].unique()
                )
            with col_f2:
                selected_internet = st.multiselect(
                    "Internet Service",
                    options=df_original['InternetService'].unique(),
                    default=df_original['InternetService'].unique()
                )
            with col_f3:
                selected_senior = st.multiselect(
                    "Senior Citizen (0=No, 1=Yes)",
                    options=[0, 1],
                    default=[0, 1]
                )
            
            # Apply custom filters
            df_data = df_data[
                (df_data['Contract'].isin(selected_contracts)) &
                (df_data['InternetService'].isin(selected_internet)) &
                (df_data['SeniorCitizen'].isin(selected_senior))
            ]

        if len(df_data) == 0:
            st.error("❌ No data matches the selected filters. Please adjust your criteria.")
        else:
            st.caption(f"Showing analysis for **{len(df_data):,}** out of **{len(df_original):,}** customers.")
            
            # KPI Cards
            total_cust = len(df_data)
            churn_rate = (df_data['Churn'] == 'Yes').mean() * 100
            avg_charge = df_data['MonthlyCharges'].mean()

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Selected Customers", f"{total_cust:,}")
            kpi2.metric("Segment Churn Rate", f"{churn_rate:.1f}%")
            kpi3.metric("Avg Monthly Charges", f"${avg_charge:.2f}")

            st.divider()

            # Row 1: Distributions
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("🎯 Churn Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_data, x='Churn', order=['No', 'Yes'], palette='Set2', ax=ax)
                ax.set_ylabel("Number of Customers")
                ax.set_xlabel("Churn Status")
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                st.subheader("📜 Churn by Contract Type")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_data, x='Contract', hue='Churn', order=sorted(df_data['Contract'].unique()), palette='Set2', ax=ax)
                ax.set_ylabel("Number of Customers")
                ax.set_xlabel("Contract Type")
                st.pyplot(fig)
                plt.close(fig)

            # Row 2: Monthly Charges & Tenure
            col_c, col_d = st.columns(2)
            with col_c:
                st.subheader("💳 Monthly Charges vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                if len(df_data['MonthlyCharges'].unique()) > 1:
                    sns.kdeplot(data=df_data, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='Set2', ax=ax)
                else:
                    sns.histplot(data=df_data, x='MonthlyCharges', hue='Churn', multiple='stack', palette='Set2', ax=ax)
                ax.set_xlabel("Monthly Charges ($)")
                st.pyplot(fig)
                plt.close(fig)

            with col_d:
                st.subheader("⏳ Tenure vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                if len(df_data['tenure'].unique()) > 1:
                    sns.kdeplot(data=df_data, x='tenure', hue='Churn', fill=True, common_norm=False, palette='Set2', ax=ax)
                else:
                    sns.histplot(data=df_data, x='tenure', hue='Churn', multiple='stack', palette='Set2', ax=ax)
                ax.set_xlabel("Tenure (Months)")
                st.pyplot(fig)
                plt.close(fig)

            st.divider()

            # Row 3: Heatmap
            st.subheader("🔗 Feature Correlation Matrix")
            df_numeric = df_data.copy()
            df_numeric['Churn'] = df_numeric['Churn'].map({'Yes': 1, 'No': 0})
            df_numeric['TotalCharges'] = pd.to_numeric(df_numeric['TotalCharges'], errors='coerce').fillna(0)
            corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
            
            corr_matrix = df_numeric[corr_cols].corr()
            if corr_matrix.isnull().values.any():
                st.info("Correlation matrix details are unavailable because some features are constant in this filtered subset.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
                st.pyplot(fig)
                plt.close(fig)

    except Exception as e:
        st.error(f"Error loading dashboard charts: {e}")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("Model: Gradient Boosting Classifier | Dataset: Telco Customer Churn")

