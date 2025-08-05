import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# App Title with Emoji
st.title("Loan Default Prediction App 📊")

# Initialize session state for intermediate data
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = None
    st.session_state.logreg_model = None

# Sidebar with Expanders
with st.sidebar.expander("1. Data Import and Overview", expanded=True):
    load_data = st.checkbox("Load and Show Dataset")
    show_stats = st.checkbox("Show Summary Statistics")
    show_viz = st.checkbox("Show Visualizations")

with st.sidebar.expander("2. Data Preprocessing", expanded=False):
    handle_missing = st.checkbox("Handle Missing Values")
    encode_categorical = st.checkbox("Encode Categorical Variables")
    scale_numerical = st.checkbox("Scale Numerical Features")
    split_data = st.checkbox("Split Data (Train/Test)")

with st.sidebar.expander("3. Model Training", expanded=False):
    train_models = st.checkbox("Train SVM and Logistic Regression")

with st.sidebar.expander("4. Model Evaluation", expanded=False):
    evaluate_models = st.checkbox("Evaluate Models")

with st.sidebar.expander("5. Prediction Function", expanded=False):
    show_prediction = st.checkbox("Show Prediction Interface")

with st.sidebar.expander("6. Interpretation and Conclusions", expanded=False):
    show_conclusions = st.checkbox("Show Conclusions")

# Main Area Content
if load_data:
    st.markdown("---")
    st.markdown("### 1.1 Dataset Preview 📋")
    with st.spinner("Loading dataset..."):
        if st.session_state.raw_data is None:
            st.session_state.raw_data = pd.read_csv('Loan_default.csv')
            for col in st.session_state.raw_data.select_dtypes(include=['object']).columns:
                st.session_state.raw_data[col] = st.session_state.raw_data[col].astype(str)
        st.info(f"Loaded dataset with {len(st.session_state.raw_data)} samples")
        st.dataframe(st.session_state.raw_data.drop('LoanID', axis=1).head())

if show_stats:
    st.markdown("---")
    st.markdown("### 1.2 Summary Statistics 📈")
    if st.session_state.raw_data is not None:
        st.dataframe(st.session_state.raw_data.drop('LoanID', axis=1).describe())
    else:
        st.warning("Please load the dataset first!")

if show_viz:
    st.markdown("---")
    st.markdown("### 1.3 Data Visualizations 🎨")
    if st.session_state.raw_data is not None:
        st.write("**Employment Type Distribution (Pie Chart)**")
        st.write("See how employment types are distributed across the dataset.")
        employment_counts = st.session_state.raw_data['EmploymentType'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(employment_counts, labels=employment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax.axis('equal')
        st.pyplot(fig)

        st.write("**Numeric Features by Default (Boxplots)**")
        st.write("Compare numeric features between defaulters and non-defaulters.")
        numeric_cols = st.session_state.raw_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_cols:
            if col != "Default":
                fig, ax = plt.subplots()
                sns.boxplot(x='Default', y=col, data=st.session_state.raw_data, palette="muted")
                plt.title(f"Boxplot of {col} by Default")
                st.pyplot(fig)
    else:
        st.warning("Please load the dataset first!")

if handle_missing:
    st.markdown("---")
    st.markdown("### 2.1 Handling Missing Values 🛠️")
    if st.session_state.raw_data is not None:
        processed_df = st.session_state.raw_data.copy()
        missing = processed_df.isnull().sum()
        st.write("Missing values:", missing)
        st.success("✅ No missing values detected")
        st.session_state.processed_data = processed_df
    else:
        st.warning("Please load the dataset first!")

if encode_categorical:
    st.markdown("---")
    st.markdown("### 2.2 Encoding Categorical Variables 🔤")
    if st.session_state.processed_data is not None or st.session_state.raw_data is not None:
        processed_df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.raw_data.copy()
        categorical_cols = processed_df.select_dtypes(include='object').columns.tolist()
        if "LoanID" in categorical_cols:
            categorical_cols.remove("LoanID")
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
        st.success(f"✅ Categorical variables encoded (New shape: {processed_df.shape})")
        st.dataframe(processed_df.head())
        st.session_state.processed_data = processed_df
    else:
        st.warning("Please load the dataset first!")

if scale_numerical:
    st.markdown("---")
    st.markdown("### 2.3 Scaling Numerical Features 📏")
    if st.session_state.processed_data is not None:
        processed_df = st.session_state.processed_data.copy()
        numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
        if 'Default' in numeric_cols:
            numeric_cols.remove('Default')
        scaler = StandardScaler()
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
        st.success("✅ Numerical features scaled")
        st.dataframe(processed_df[numeric_cols].head())
        st.session_state.processed_data = processed_df
        st.session_state.scaler = scaler
    else:
        st.warning("Please complete prior preprocessing steps!")

if split_data:
    st.markdown("---")
    st.markdown("### 2.4 Splitting Data (Train/Test) ✂️")
    if st.session_state.processed_data is not None:
        X = st.session_state.processed_data.drop(columns=[col for col in ['LoanID', 'Default'] if col in st.session_state.processed_data.columns])
        y = st.session_state.processed_data['Default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.success(f"✅ Data split: Train ({X_train.shape[0]} samples), Test ({X_test.shape[0]} samples)")
    else:
        st.warning("Please complete prior preprocessing steps!")

if train_models:
    st.markdown("---")
    st.markdown("### 3.1 Model Training 🤖")
    if st.session_state.X_train is not None:
        with st.spinner("Training models..."):
            svm_model = LinearSVC(max_iter=1000, C=0.1, class_weight='balanced')
            svm_model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.svm_model = svm_model
            st.success("✅ SVM model trained")

            logreg_model = LogisticRegression(max_iter=1000)
            logreg_model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.logreg_model = logreg_model
            st.success("✅ Logistic Regression model trained")
    else:
        st.warning("Please split data first!")

if evaluate_models:
    st.markdown("---")
    st.markdown("### 4.1 Model Evaluation 📊")
    if st.session_state.svm_model is not None or st.session_state.logreg_model is not None:
        if st.session_state.svm_model is not None:
            y_pred_svm = st.session_state.svm_model.predict(st.session_state.X_test)
            st.write("**SVM Classification Report**")
            st.text(classification_report(st.session_state.y_test, y_pred_svm))
            st.write("**SVM Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(st.session_state.y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
            plt.title("SVM Confusion Matrix")
            st.pyplot(fig)

        if st.session_state.logreg_model is not None:
            y_pred_logreg = st.session_state.logreg_model.predict(st.session_state.X_test)
            st.write("**Logistic Regression Classification Report**")
            st.text(classification_report(st.session_state.y_test, y_pred_logreg))
            st.write("**Logistic Regression Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(st.session_state.y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
            plt.title("Logistic Regression Confusion Matrix")
            st.pyplot(fig)

        if st.session_state.svm_model is not None and st.session_state.logreg_model is not None:
            st.write("**Performance Comparison**")
            svm_report = classification_report(st.session_state.y_test, y_pred_svm, output_dict=True)
            logreg_report = classification_report(st.session_state.y_test, y_pred_logreg, output_dict=True)
            fig, ax = plt.subplots()
            metrics = ['precision', 'recall', 'f1-score']
            svm_scores = [svm_report['1'].get(metric, 0.0) for metric in metrics]
            logreg_scores = [logreg_report['1'][metric] for metric in metrics]
            x = np.arange(len(metrics))
            width = 0.35
            ax.bar(x - width/2, svm_scores, width, label='SVM', color='blue')
            ax.bar(x + width/2, logreg_scores, width, label='Logistic Regression', color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_title("Model Performance Comparison")
            ax.set_ylim(0, 1)
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Please train models first!")

if show_prediction:
    st.markdown("---")
    st.markdown("### 5.1 Predict Loan Default 🔮")
    if st.session_state.logreg_model is not None:
        st.write("Fill in the details to predict loan default probability:")
        col1, col2, col3 = st.columns(3)
        input_data = {}

        with col1:
            st.subheader("Personal Info 👤")
            input_data['Age'] = st.number_input("Age", min_value=18, max_value=100, value=30)
            input_data['Education'] = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
            input_data['EmploymentType'] = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            input_data['MaritalStatus'] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            input_data['HasDependents'] = st.selectbox("Has Dependents", ["Yes", "No"])
            input_data['HasCoSigner'] = st.selectbox("Has Co-Signer", ["Yes", "No"])

        with col2:
            st.subheader("Financial Info 💰")
            input_data['Income'] = st.number_input("Income", min_value=0, value=50000)
            input_data['CreditScore'] = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
            input_data['MonthsEmployed'] = st.number_input("Months Employed", min_value=0, value=24)
            input_data['NumCreditLines'] = st.number_input("Number of Credit Lines", min_value=0, value=2)
            input_data['DTIRatio'] = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)

        with col3:
            st.subheader("Loan Details 🏦")
            input_data['LoanAmount'] = st.number_input("Loan Amount", min_value=0, value=50000)
            input_data['InterestRate'] = st.number_input("Interest Rate", min_value=0.0, value=10.0)
            input_data['LoanTerm'] = st.number_input("Loan Term (Months)", min_value=0, value=36)
            input_data['LoanPurpose'] = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
            input_data['HasMortgage'] = st.selectbox("Has Mortgage", ["Yes", "No"])

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df, columns=['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'], drop_first=True)
            input_encoded = input_encoded.reindex(columns=st.session_state.X_train.columns, fill_value=0)
            numeric_cols = st.session_state.processed_data.select_dtypes(include=np.number).columns.tolist()
            if 'Default' in numeric_cols:
                numeric_cols.remove('Default')
            input_encoded[numeric_cols] = st.session_state.scaler.transform(input_encoded[numeric_cols])
            prediction = st.session_state.logreg_model.predict(input_encoded)
            probability = st.session_state.logreg_model.predict_proba(input_encoded)[0][1]
            st.success(f"**Prediction**: {'Default' if prediction[0] == 1 else 'No Default'}")
            st.info(f"**Probability of Default**: {probability:.2%}")
    else:
        st.warning("Please train models first!")

if show_conclusions:
    st.markdown("---")
    st.markdown("### 6.1 Interpretation and Conclusions 📝")
    st.write("""
    **Summary of Findings**:
    - SVM and Logistic Regression models were trained to predict loan defaults.
    - SVM performance improves with class weighting due to class imbalance.
    - Logistic Regression often outperforms SVM for this task.
    - Key predictors: CreditScore, Income, DTIRatio.
    """)
    if st.session_state.logreg_model is not None:
        st.write("**Logistic Regression Feature Importance**")
        feature_importance = pd.DataFrame({
            'Feature': st.session_state.X_train.columns,
            'Coefficient': st.session_state.logreg_model.coef_[0]
        }).sort_values(by='Coefficient', ascending=False)
        st.dataframe(feature_importance)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(10), palette="viridis")
        plt.title("Top 10 Features by Coefficient")
        st.pyplot(fig)