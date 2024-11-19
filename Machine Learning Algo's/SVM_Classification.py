import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import io
import ydata_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  
    df.drop_duplicates(inplace=True)
    
    # Example: Convert data types (if necessary)
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype('int')
    
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['new_feature'] = df['feature1'] * df['feature2']
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    return df

st.title("Support Vector Classifier")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Dataset Information:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.write("Statistical Summary:")
    st.write(data.describe())

    st.write("Dataset Preview:")
    st.write(data.head())

    data = preprocess_data(data)
    st.write("General Dataset preprocessing is completed")
    
    st.write("Statistical Summary After Pre-processing:")
    st.write(data.describe())
    
    st.write("Generating Pandas Profiling Report...")
    profile = data.profile_report()
    st_profile_report(profile)

    # Select target and feature variables
    st.write("Select Feature and Target Variables:")
    columns_with_all = ["All"] + list(data.columns)
    target_columns = st.multiselect("Select target variable(s) [Y - variables]", data.columns)
    feature_columns = st.multiselect("Select feature variable(s) (default is 'All') [X - Variables]", columns_with_all, default="All")

    if "All" in feature_columns:
        feature_columns = [col for col in data.columns if col not in target_columns]

    if not target_columns:
        st.warning("Please select at least one target variable.")
    else:
        X = data[feature_columns]
        y = data[target_columns]

        # Select variables for encoding from X
        st.write("Select variables from X for One-Hot Encoding:")
        selected_cat_columns = st.multiselect("Select columns to convert to categorical and one-hot encode", X.columns)

        for col in selected_cat_columns:
            X[col] = X[col].astype('category')

        if selected_cat_columns:
            X = pd.get_dummies(X, columns=selected_cat_columns, drop_first=True)
            st.write("One-Hot Encoded Feature Variables:")
            st.write(X.head())

        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # SVM Hyperparameters
        st.sidebar.header("Model Hyperparameters")
        C = st.sidebar.slider("C (Regularization parameter)", 0.01, 100.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], index=2)  # Default is 'rbf'
        degree = st.sidebar.slider("Degree (for 'poly' kernel)", 2, 5, 3)
        gamma = st.sidebar.selectbox("Gamma (Kernel Coefficient)", ['scale', 'auto'], index=0)

        # Train SVM Classifier
        st.write("Training the Support Vector Classifier...")
        svc_model = SVC(C=C, kernel=kernel, degree=degree if kernel == 'poly' else 3, gamma=gamma)
        svc_model.fit(X_train, y_train)

        # Make predictions on training, validation, and test sets
        y_train_pred = svc_model.predict(X_train)
        y_val_pred = svc_model.predict(X_val)
        y_test_pred = svc_model.predict(X_test)

        # Evaluate the model
        def evaluate_model(y_true, y_pred, set_name=""):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y_true, y_pred, average='weighted')
            st.write(f"**{set_name} Set Performance**")
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Precision: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1 Score: {f1}")
            st.write("-----")

        # Evaluate on training, validation, and test sets
        evaluate_model(y_train, y_train_pred, "Training")
        evaluate_model(y_val, y_val_pred, "Validation")
        evaluate_model(y_test, y_test_pred, "Test")

        # Confusion matrix visualization
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm")
        st.pyplot(plt)

        # Perform Grid Search for Hyperparameters
        st.write("Performing Grid Search for Hyperparameters...")

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf'],
            'degree': [2, 3, 4] if 'poly' in kernel else [3],
            'gamma': ['scale', 'auto']
        }

        grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)

        st.write(f"Best Hyperparameters: {grid_search.best_params_}")

        results = pd.DataFrame(grid_search.cv_results_)
        st.write("Grid Search Results:")
        st.write(results)

else:
    st.write("Please upload a CSV file to begin.")
