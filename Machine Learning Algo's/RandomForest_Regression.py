import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import seaborn as sns
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  
    df.drop_duplicates(inplace=True)
    
    # Convert data types (example)
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype('int')
    
    # Create new features (example)
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['new_feature'] = df['feature1'] * df['feature2']
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df


st.title("Random Forest Regressor")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Dataset Information:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)  # Show DataFrame info and description

    st.write("Statistical Summary:")
    st.write(data.describe())

    st.write("Dataset Preview:")
    st.write(data.head())

    data = preprocess_data(data)
    st.write("General Dataset preprocessing is completed")  # Apply preprocessing in one line
    
    st.write("Statistical Summary After Pre-processing:")
    st.write(data.describe())

    st.write("Generating Pandas Profiling Report...")  # Generate a pandas profiling report
    profile = data.profile_report()
    st_profile_report(profile)

    # Select target and feature variables (user selects)
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

        # Select variables for encoding from X
        st.write("Select variables from X for One-Hot Encoding:")
        selected_cat_columns = st.multiselect("Select columns to convert to categorical and one-hot encode", X.columns)

        for col in selected_cat_columns:
            X[col] = X[col].astype('category')

        # One-Hot Encode the selected categorical columns
        if selected_cat_columns:
            X = pd.get_dummies(X, columns=selected_cat_columns, drop_first=True)
            st.write("One-Hot Encoded Feature Variables:")
            st.write(X.head())
        else:
            st.warning("No categorical columns selected for encoding.")

        # Split data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Hyperparameter tuning options for Random Forest
        st.sidebar.header("Random Forest Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        max_features = st.sidebar.selectbox("Max Features", options=['sqrt', 'log2', None])

        # Train Random Forest Regressor
        st.write("Training the Random Forest Regressor...")
        rf_reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                    max_features=max_features, random_state=42)
        rf_reg.fit(X_train, y_train)


        # Make predictions on training, validation, and test sets
        y_train_pred = rf_reg.predict(X_train)
        y_val_pred = rf_reg.predict(X_val)
        y_test_pred = rf_reg.predict(X_test)

        # Evaluate the model (RMSE and MAE) and Correlation for Train, Val, and Test sets
        def evaluate_model(y_true, y_pred, set_name=""):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            st.write(f"**{set_name} Set Performance**")
            st.write(f"RMSE: {rmse}")
            st.write(f"MAE: {mae}")
            st.write(f"Correlation: {correlation}")
            st.write("-----")

        for target in target_columns:
            evaluate_model(y_train[target], y_train_pred, "Training")
            evaluate_model(y_val[target], y_val_pred, "Validation")
            evaluate_model(y_test[target], y_test_pred, "Test")

        # Perform Grid Search for RMSE and MSE
        st.write("Running Grid Search for RMSE and MSE...")

        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt']
        }

        grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)

        st.write(f"Best Hyperparameters: {grid_search.best_params_}")

        # Plot RMSE and MSE over Grid Search
        results = pd.DataFrame(grid_search.cv_results_)
        results['mean_test_rmse'] = np.sqrt(-results['mean_test_score'])
        results['mean_test_mse'] = -results['mean_test_score']

        st.write("RMSE Plot:")
        plt.figure(figsize=(10, 5))
        plt.plot(results.index, results['mean_test_rmse'], label="RMSE")
        plt.xlabel('Grid Search Run')
        plt.ylabel('RMSE')
        plt.legend()
        st.pyplot(plt)

        st.write("MSE Plot:")
        plt.figure(figsize=(10, 5))
        plt.plot(results.index, results['mean_test_mse'], label="MSE")
        plt.xlabel('Grid Search Run')
        plt.ylabel('MSE')
        plt.legend()
        st.pyplot(plt)

        # Feature importance
        st.write("Feature importance:")
        feature_importances = pd.Series(rf_reg.feature_importances_, index=X_train.columns)
        fig, ax = plt.subplots()
        feature_importances.nlargest(10).plot(kind='barh', ax=ax)
        st.pyplot(fig)

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
