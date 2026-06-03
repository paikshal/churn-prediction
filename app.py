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
    st.markdown("This dashboard displays insights and patterns from the Telco Customer Churn dataset dynamically based on your filter selection.")

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

            # Visualizations Selector
            st.subheader("📈 Visualization Center")
            eda_option = st.selectbox(
                "Choose Analysis Visualizations (Select to update plots):",
                options=[
                    "Overall Churn Distribution",
                    "Numeric Features vs Churn (Boxplots & KDE)",
                    "Tenure Group Analysis",
                    "Categorical Features Churn Rate",
                    "Service Features Churn Rate",
                    "Correlation Matrix & Heatmap",
                    "Pairplot (Key Numeric Features)"
                ]
            )

            colors_map = {'No': '#43d394', 'Yes': '#ff4d6d'}

            # 1. Overall Churn Distribution
            if eda_option == "Overall Churn Distribution":
                churn_counts = df_data['Churn'].value_counts()
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.subheader("🎯 Churn Count Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x='Churn', data=df_data, palette=colors_map, order=['No', 'Yes'], ax=ax)
                    for p in ax.patches:
                        h = int(p.get_height())
                        ax.annotate(f'{h}', (p.get_x() + 0.3, h + len(df_data)*0.01))
                    ax.set_ylabel("Number of Customers")
                    ax.set_xlabel("Churn Status")
                    st.pyplot(fig)
                    plt.close(fig)

                with col_c2:
                    st.subheader("🍕 Churn Proportion")
                    if len(churn_counts) > 0:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        pie_colors = [colors_map[idx] for idx in churn_counts.index if idx in colors_map]
                        ax.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=pie_colors, startangle=90)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.write("No data to show pie chart.")

                st.info("ℹ️ **Class Imbalance Note:** Original dataset holds a ~73:27 ratio. In filtered segments, keep this ratio in mind when training/testing models.")

            # 2. Numeric Features vs Churn
            elif eda_option == "Numeric Features vs Churn (Boxplots & KDE)":
                num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
                df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')
                df_data['TotalCharges'] = df_data['TotalCharges'].fillna(df_data['TotalCharges'].median())
                
                st.markdown("#### Boxplots (Value Range & Outliers)")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for i, col in enumerate(num_cols):
                    sns.boxplot(x='Churn', y=col, data=df_data, palette=colors_map, order=['No', 'Yes'], ax=axes[i])
                    axes[i].set_title(f'{col} vs Churn')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("#### KDE Plots (Probability Density Shape)")
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for i, col in enumerate(num_cols):
                    if len(df_data[col].unique()) > 1:
                        for churn_val, color, label in [('No', '#43d394', 'Retained'), ('Yes', '#ff4d6d', 'Churned')]:
                            subset = df_data[df_data['Churn'] == churn_val][col]
                            if len(subset.unique()) > 1:
                                subset.plot(kind='kde', ax=axes[i], color=color, label=label, linewidth=2)
                        axes[i].set_title(f'{col} — KDE Density')
                        axes[i].legend()
                    else:
                        axes[i].text(0.5, 0.5, "Single distinct value. Cannot plot density.", ha='center', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("#### Mean Values by Churn Status")
                st.dataframe(df_data.groupby('Churn')[num_cols].mean().round(2))

            # 3. Tenure Group Analysis
            elif eda_option == "Tenure Group Analysis":
                st.subheader("⏳ Churn Behavior by Tenure Range")

                def tenure_group_labels(t):
                    if t <= 12:  return '0-12 Mo'
                    elif t <= 24: return '13-24 Mo'
                    elif t <= 48: return '25-48 Mo'
                    else:         return '48+ Mo'

                df_tg = df_data.copy()
                df_tg['Tenure_Group'] = df_tg['tenure'].apply(tenure_group_labels)
                order = ['0-12 Mo', '13-24 Mo', '25-48 Mo', '48+ Mo']

                try:
                    tg = pd.crosstab(df_tg['Tenure_Group'], df_tg['Churn'], normalize='index') * 100
                    for grp in order:
                        if grp not in tg.index:
                            tg.loc[grp] = [0.0, 0.0]
                    tg = tg.loc[order]
                    
                    col_t1, col_t2 = st.columns([1, 2])
                    with col_t1:
                        st.markdown("#### Churn Rate % Table")
                        st.dataframe(tg.round(1))
                    with col_t2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        tg['Yes'].plot(kind='bar', color='#ff4d6d', edgecolor='none', ax=ax)
                        ax.set_title('Churn Rate by Tenure Group (%)')
                        ax.set_ylabel('Churn %')
                        plt.xticks(rotation=0)
                        overall_avg = (df_data['Churn'] == 'Yes').mean() * 100
                        ax.axhline(overall_avg, linestyle='--', color='gray', alpha=0.6, label=f'Overall segment avg ({overall_avg:.1f}%)')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Error compiling tenure group chart: {e}")

            # 4. Categorical Features Churn Rate
            elif eda_option == "Categorical Features Churn Rate":
                cat_cols_list = ['Contract', 'InternetService', 'PaymentMethod', 'SeniorCitizen', 'PaperlessBilling']
                selected_cat = st.selectbox("Select Categorical Column to Analyze:", options=cat_cols_list)

                try:
                    df_cat = df_data.copy()
                    if selected_cat == 'SeniorCitizen':
                        df_cat['SeniorCitizen'] = df_cat['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

                    ct = pd.crosstab(df_cat[selected_cat], df_cat['Churn'], normalize='index') * 100
                    
                    if 'Yes' in ct.columns:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ct['Yes'].sort_values().plot(kind='barh', color='#ff4d6d', edgecolor='none', ax=ax)
                        overall_avg = (df_data['Churn'] == 'Yes').mean() * 100
                        ax.axvline(overall_avg, linestyle='--', color='gray', alpha=0.6, label=f'Overall segment avg ({overall_avg:.1f}%)')
                        ax.set_title(f'Churn Rate by {selected_cat} (%)')
                        ax.set_xlabel('Churn %')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        st.markdown("#### Detail Table")
                        st.dataframe(ct.round(1))
                    else:
                        st.warning("No churned customers found in the selected subset for this visualization.")
                except Exception as e:
                    st.error(f"Error compiling categorical charts: {e}")

            # 5. Service Features Churn Rate
            elif eda_option == "Service Features Churn Rate":
                service_cols_list = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
                df_serv = df_data.copy()
                df_serv['Churn_num'] = df_serv['Churn'].map({'Yes': 1, 'No': 0})
                
                churn_rates = {}
                for col in service_cols_list:
                    if col in df_serv.columns:
                        subset_yes = df_serv[df_serv[col] == 'Yes']
                        if len(subset_yes) > 0:
                            rate = subset_yes['Churn_num'].mean() * 100
                            churn_rates[col] = round(rate, 1)
                        else:
                            churn_rates[col] = 0.0

                if churn_rates:
                    churn_series = pd.Series(churn_rates).sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    churn_series.plot(kind='barh', color='#4cc9f0', edgecolor='none', ax=ax)
                    overall_avg = (df_data['Churn'] == 'Yes').mean() * 100
                    ax.axvline(overall_avg, linestyle='--', color='#ff4d6d', alpha=0.7, label=f'Overall segment avg ({overall_avg:.1f}%)')
                    ax.set_title('Churn Rate when Service Option is Active (Yes)')
                    ax.set_xlabel('Churn %')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.markdown("#### Churn Rate breakdown by service:")
                    st.json(churn_rates)

            # 6. Correlation Matrix & Heatmap
            elif eda_option == "Correlation Matrix & Heatmap":
                try:
                    df_num = df_data.copy()
                    service_cols_bin = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
                    df_num['TotalServices'] = (df_num[service_cols_bin] == 'Yes').sum(axis=1)
                    df_num['Churn_num'] = df_num['Churn'].map({'Yes': 1, 'No': 0})
                    
                    if 'gender' in df_num.columns:
                        df_num['gender'] = df_num['gender'].map({'Male': 1, 'Female': 0})
                    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                        if col in df_num.columns:
                            df_num[col] = df_num[col].map({'Yes': 1, 'No': 0})
                    
                    df_num['TotalCharges'] = pd.to_numeric(df_num['TotalCharges'], errors='coerce').fillna(0)
                    numeric_df = df_num.select_dtypes(include=[np.number])
                    corr = numeric_df.corr()
                    
                    if corr.isnull().values.any():
                        st.info("Correlation matrix details are unavailable because some features are constant in this filtered subset.")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.4, vmin=-1, vmax=1, ax=ax)
                        ax.set_title('Correlation Matrix (Lower Triangle)')
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        if 'Churn_num' in corr.columns:
                            st.markdown("#### Feature Correlation with Churn:")
                            corr_churn = corr['Churn_num'].drop('Churn_num', errors='ignore').sort_values(key=abs, ascending=False)
                            st.dataframe(corr_churn.round(3).rename("Correlation with Churn"))
                except Exception as e:
                    st.error(f"Error compiling correlation heatmap: {e}")

            # 7. Pairplot (Key Numeric Features)
            elif eda_option == "Pairplot (Key Numeric Features)":
                try:
                    df_pair = df_data.copy()
                    df_pair['TotalCharges'] = pd.to_numeric(df_pair['TotalCharges'], errors='coerce').fillna(0)
                    plot_df = df_pair[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].dropna()
                    
                    if len(plot_df) > 1:
                        fig = sns.pairplot(plot_df, hue='Churn', palette={'No': '#43d394', 'Yes': '#ff4d6d'},
                                           plot_kws={'alpha': 0.4, 's': 15}, diag_kind='kde')
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough data to construct pairplot.")
                except Exception as e:
                    st.error(f"Error compiling pairplot: {e}")

    except Exception as e:
        st.error(f"Error loading dashboard charts: {e}")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("Model: Gradient Boosting Classifier | Dataset: Telco Customer Churn")

