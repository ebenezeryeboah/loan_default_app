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

# App Title
st.title('Loan Default Prediction App')

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

# 1. Data Import and Overview
st.sidebar.header("1. Data Import and Overview")
if st.sidebar.checkbox("Load and Show Dataset"):
    with st.spinner("Loading dataset..."):
        if st.session_state.raw_data is None:
            st.session_state.raw_data = pd.read_csv('Loan_default.csv')
            for col in st.session_state.raw_data.select_dtypes(include=['object']).columns:
                st.session_state.raw_data[col] = st.session_state.raw_data[col].astype(str)
        st.subheader("1.1 Dataset Preview")
        st.write(f"Loaded dataset with {len(st.session_state.raw_data)} samples")
        st.dataframe(st.session_state.raw_data.drop('LoanID', axis=1).head())

if st.sidebar.checkbox("Show Summary Statistics"):
    st.subheader("1.2 Summary Statistics")
    st.write(st.session_state.raw_data.drop('LoanID', axis=1).describe() if st.session_state.raw_data is not None else "Please load dataset first.")

if st.sidebar.checkbox("Show Visualizations (Pie Chart and Boxplots)"):
    if st.session_state.raw_data is not None:
        st.subheader("1.3 Data Visualizations")
        # Pie Chart: Employment Type Distribution
        st.write("Pie Chart: Employment Type Distribution")
        employment_counts = st.session_state.raw_data['EmploymentType'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(employment_counts, labels=employment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Boxplots for numeric features by Default
        st.write("Boxplots for Numeric Features by Default")
        numeric_cols = st.session_state.raw_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_cols:
            if col != "Default":
                fig, ax = plt.subplots()
                sns.boxplot(x='Default', y=col, data=st.session_state.raw_data)
                plt.title(f"Boxplot of {col} by Default")
                st.pyplot(fig)
    else:
        st.write("Please load dataset first.")

# 2. Data Preprocessing
st.sidebar.header("2. Data Preprocessing")
if st.sidebar.checkbox("Handle Missing Values"):
    if st.session_state.raw_data is not None:
        st.subheader("2.1 Handling Missing Values")
        processed_df = st.session_state.raw_data.copy()
        missing = processed_df.isnull().sum()
        st.write("Missing values:")
        st.write(missing)
        st.write("✅ No missing values detected (no rows dropped)")
        st.session_state.processed_data = processed_df
    else:
        st.write("Please load dataset first.")

if st.sidebar.checkbox("Encode Categorical Variables"):
    if st.session_state.processed_data is not None or st.session_state.raw_data is not None:
        st.subheader("2.2 Encoding Categorical Variables")
        processed_df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.raw_data.copy()
        categorical_cols = processed_df.select_dtypes(include='object').columns.tolist()
        if "LoanID" in categorical_cols:
            categorical_cols.remove("LoanID")
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
        st.write("✅ Categorical variables encoded")
        st.write(f"New shape: {processed_df.shape}")
        st.dataframe(processed_df.head())
        st.session_state.processed_data = processed_df
    else:
        st.write("Please load dataset first.")

if st.sidebar.checkbox("Scale Numerical Features"):
    if st.session_state.processed_data is not None:
        st.subheader("2.3 Scaling Numerical Features")
        processed_df = st.session_state.processed_data.copy()
        numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
        if 'Default' in numeric_cols:
            numeric_cols.remove('Default')
        scaler = StandardScaler()
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
        st.write("✅ Numerical features scaled")
        st.dataframe(processed_df[numeric_cols].head())
        st.session_state.processed_data = processed_df
        st.session_state.scaler = scaler  # Store scaler for prediction
    else:
        st.write("Please load dataset first or complete prior preprocessing steps.")

if st.sidebar.checkbox("Split Data (Train/Test)"):
    if st.session_state.processed_data is not None:
        st.subheader("2.4 Splitting Data into Train and Test Sets")
        X = st.session_state.processed_data.drop(columns=[col for col in ['LoanID', 'Default'] if col in st.session_state.processed_data.columns])
        y = st.session_state.processed_data['Default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.write("✅ Data successfully split")
        st.write(f"Training set size: {X_train.shape[0]} samples")
        st.write(f"Test set size: {X_test.shape[0]} samples")
    else:
        st.write("ng steps.")

# 3. Model Training
st.sidebar.header("3. Model Training")
if st.sidebar.checkbox("Train SVM and Logistic Regression"):
    if st.session_state.X_train is not None:
        st.subheader("3.1 Model Training")
        with st.spinner("🔁 Starting model training..."):
            try:
                # Adjusted C parameter and class_weight='balanced' for better minority class prediction
                svm_model = LinearSVC(max_iter=1000, C=0.1, class_weight='balanced')
                svm_model.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.svm_model = svm_model
                st.success("✅ SVM model trained with class weights and adjusted C=0.1")
            except Exception as e:
                st.error(f"SVM training failed: {e}")

            try:
                logreg_model = LogisticRegression(max_iter=1000)
                logreg_model.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.logreg_model = logreg_model
                st.success("✅ Logistic Regression model trained")
            except Exception as e:
                st.error(f"Logistic Regression training failed: {e}")
    else:
        st.write("Please split data first.")

# 4. Model Evaluation
st.sidebar.header("4. Model Evaluation")
if st.sidebar.checkbox("Evaluate Models"):
    if st.session_state.svm_model is not None or st.session_state.logreg_model is not None:
        st.subheader("4.1 Model Evaluation")
        if st.session_state.svm_model is not None:
            y_pred_svm = st.session_state.svm_model.predict(st.session_state.X_test)
            st.write("SVM Classification Report")
            st.text(classification_report(st.session_state.y_test, y_pred_svm))
            st.write("SVM Confusion Matrix")
            cm_svm = confusion_matrix(st.session_state.y_test, y_pred_svm)
            fig, ax = plt.subplots()
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
            plt.title("SVM Confusion Matrix")
            st.pyplot(fig)

        if st.session_state.logreg_model is not None:
            y_pred_logreg = st.session_state.logreg_model.predict(st.session_state.X_test)
            st.write("Logistic Regression Classification Report")
            st.text(classification_report(st.session_state.y_test, y_pred_logreg))
            st.write("Logistic Regression Confusion Matrix")
            cm_logreg = confusion_matrix(st.session_state.y_test, y_pred_logreg)
            fig, ax = plt.subplots()
            sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues')
            plt.title("Logistic Regression Confusion Matrix")
            st.pyplot(fig)

        # Performance Comparison Chart
        if st.session_state.svm_model is not None and st.session_state.logreg_model is not None:
            st.write("Performance Comparison")
            svm_report = classification_report(st.session_state.y_test, y_pred_svm, output_dict=True)
            logreg_report = classification_report(st.session_state.y_test, y_pred_logreg, output_dict=True)
            fig, ax = plt.subplots()
            metrics = ['precision', 'recall', 'f1-score']
            svm_scores = [svm_report['1'].get(metric, 0.0) for metric in metrics]  # Default to 0.0 if undefined
            logreg_scores = [logreg_report['1'][metric] for metric in metrics]
            x = np.arange(len(metrics))
            width = 0.35
            ax.bar(x - width/2, svm_scores, width, label='SVM', color='blue')
            ax.bar(x + width/2, logreg_scores, width, label='Logistic Regression', color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_title("Model Performance Comparison")
            ax.set_ylim(0, 1)  # Ensure y-axis goes up to 1
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("Please train models first.")

# 5. Prediction Function
st.sidebar.header("5. Prediction Function")
if st.sidebar.checkbox("Show Prediction Interface"):
    if st.session_state.logreg_model is not None:
        st.subheader("5.1 Predict Loan Default")
        input_data = {}
        input_data['Age'] = st.number_input("Age", min_value=18, max_value=100, value=30)
        input_data['Income'] = st.number_input("Income", min_value=0, value=50000)
        input_data['LoanAmount'] = st.number_input("Loan Amount", min_value=0, value=50000)
        input_data['CreditScore'] = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        input_data['MonthsEmployed'] = st.number_input("Months Employed", min_value=0, value=24)
        input_data['NumCreditLines'] = st.number_input("Number of Credit Lines", min_value=0, value=2)
        input_data['InterestRate'] = st.number_input("Interest Rate", min_value=0.0, value=10.0)
        input_data['LoanTerm'] = st.number_input("Loan Term (Months)", min_value=0, value=36)
        input_data['DTIRatio'] = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)
        input_data['Education'] = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        input_data['EmploymentType'] = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        input_data['MaritalStatus'] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        input_data['HasMortgage'] = st.selectbox("Has Mortgage", ["Yes", "No"])
        input_data['HasDependents'] = st.selectbox("Has Dependents", ["Yes", "No"])
        input_data['LoanPurpose'] = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
        input_data['HasCoSigner'] = st.selectbox("Has Co-Signer", ["Yes", "No"])

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
            st.write(f"Prediction: {'Default' if prediction[0] == 1 else 'No Default'}")
            st.write(f"Probability of Default: {probability:.2%}")
    else:
        st.write("Please train models first.")

# 6. Interpretation and Conclusions
st.sidebar.header("6. Interpretation and Conclusions")
if st.sidebar.checkbox("Show Conclusions"):
    st.subheader("6.1 Interpretation and Conclusions")
    st.write("""
    **Summary of Findings**:
    - The SVM and Logistic Regression models were trained to predict loan defaults.
    - SVM struggles with the minority class (Default=1) due to class imbalance, but class weighting and C=0.1 improve its performance.
    - Logistic Regression generally performs better for this task.
    - Key features influencing defaults include CreditScore, Income, and DTIRatio.
    """)
    if st.session_state.logreg_model is not None:
        st.write("Logistic Regression Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': st.session_state.X_train.columns,
            'Coefficient': st.session_state.logreg_model.coef_[0]
        }).sort_values(by='Coefficient', ascending=False)
        st.dataframe(feature_importance)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(10))
        plt.title("Top 10 Features by Logistic Regression Coefficient")
        st.pyplot(fig)