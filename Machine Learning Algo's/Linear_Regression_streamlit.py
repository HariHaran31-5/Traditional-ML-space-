import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  # Forward-fill missing values
    df.drop_duplicates(inplace=True)
    
    # Convert data types (example)
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype('int')
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)  # Use drop_first to avoid dummy variable trap

    return df

st.title("Linear Regression with Hyperparameter Tuning")

# File upload
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
        # Define features and target variable
        X = data[feature_columns]
        y = data[target_columns]

        # Optional: Feature scaling
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Sidebar for Linear Regression Hyperparameters
        st.sidebar.header("Hyperparameter Tuning")
        fit_intercept = st.sidebar.selectbox("Fit Intercept", [True, False], index=0)
        n_jobs = st.sidebar.number_input("Number of Jobs (parallel processing)", min_value=1, max_value=8, step=1, value=1)
        positive = st.sidebar.selectbox("Positive Coefficients Only", [True, False], index=1)

        # Hyperparameter grid for GridSearchCV (without the 'normalize' parameter)
        param_grid = {
            'fit_intercept': [fit_intercept],
            'n_jobs': [n_jobs],
            'positive': [positive]
        }

        # Split data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Train Linear Regression model
        st.write("Training the Linear Regression Model...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Grid Search for hyperparameter tuning
        st.write("Performing GridSearchCV...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Best parameters from Grid Search
        best_params = grid_search.best_params_
        st.write(f"Best Parameters: {best_params}")

        # Train the model with the best parameters
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluate the model (RMSE and MAE) and Correlation for Train, Val, and Test sets
        def evaluate_model(y_true, y_pred, set_name=""):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            correlation = np.corrcoef(y_true.T, y_pred.T)[0, 1]
            st.write(f"**{set_name} Set Performance**")
            st.write(f"RMSE: {rmse}")
            st.write(f"MAE: {mae}")
            st.write(f"Correlation: {correlation}")
            st.write("-----")

        for target in target_columns:
            evaluate_model(y_train[target], y_train_pred, "Training")
            evaluate_model(y_val[target], y_val_pred, "Validation")
            evaluate_model(y_test[target], y_test_pred, "Test")

        # Feature importance visualization (coefficients)
        st.write("Feature Importance (Coefficients):")
        if len(target_columns) == 1:
            # Single target case
            coeff = pd.DataFrame(model.coef_.reshape(-1, 1), index=X.columns, columns=['Coefficient'])
            st.write(coeff)
        else:
            # Multiple target case (coef_ will have a shape of (n_targets, n_features))
            for idx, target in enumerate(target_columns):
                coeff = pd.DataFrame(model.coef_[idx], index=X.columns, columns=[f'Coefficient for {target}'])
                st.write(f"Coefficients for target: {target}")
                st.write(coeff)

        # Correlation Matrix with Heatmap
        st.write("Correlation Heatmap:")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)

        # Pairplot (Scatterplot Matrix)
        st.write("Pairplot:")
        selected_columns_for_pairplot = st.multiselect("Select columns for Pairplot", data.columns)
        if len(selected_columns_for_pairplot) > 1:
            sns.pairplot(data[selected_columns_for_pairplot])
            st.pyplot()

        # Feature Distribution (Histograms)
        st.write("Feature Distributions:")
        selected_columns_for_hist = st.multiselect("Select columns to visualize distributions", data.columns)
        for col in selected_columns_for_hist:
            plt.figure(figsize=(7, 5))
            sns.histplot(data[col], kde=True, bins=30)
            st.pyplot(plt)

else:
    st.write("Please upload a CSV file to begin.")
