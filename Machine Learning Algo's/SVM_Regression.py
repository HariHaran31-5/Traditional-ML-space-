import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import io
import seaborn as sns

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  
    df.drop_duplicates(inplace=True)
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype('int')
    
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['new_feature'] = df['feature1'] * df['feature2']
    
    df = pd.get_dummies(df, drop_first=True)
    return df

st.title("Support Vector Regressor with User-Selected Grid Search")

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

        st.write("Select variables from X for One-Hot Encoding:")
        selected_cat_columns = st.multiselect("Select columns to convert to categorical and one-hot encode", X.columns)

        for col in selected_cat_columns:
            X[col] = X[col].astype('category')

        if selected_cat_columns:
            X = pd.get_dummies(X, columns=selected_cat_columns, drop_first=True)
            st.write("One-Hot Encoded Feature Variables:")
            st.write(X.head())
        else:
            st.warning("You don't have any categorical columns for encoding.")

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Hyperparameter Input from User
        st.sidebar.header("Model Hyperparameters")

        # Allow user input for C values
        C_range = st.sidebar.text_input("Enter range of C values (comma-separated, e.g., 0.1,1,10)", "0.1,1,10")
        C_range = [float(c) for c in C_range.split(',')]  # Convert to float list

        # Allow user to select kernels
        kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
        kernel_selected = st.sidebar.multiselect("Select kernels to try", kernel_options, default='rbf')

        # Degree only if 'poly' is selected
        degree_range = [3]  # Default for non-poly
        if 'poly' in kernel_selected:
            degree_range = st.sidebar.text_input("Enter degrees for 'poly' kernel (comma-separated, e.g., 2,3,4)", "2,3,4")
            degree_range = [int(d) for d in degree_range.split(',')]  # Convert to int list

        # Allow user input for epsilon values
        epsilon_range = st.sidebar.text_input("Enter range of epsilon values (comma-separated, e.g., 0.1,0.2)", "0.1,0.2")
        epsilon_range = [float(eps) for eps in epsilon_range.split(',')]  # Convert to float list

        # Choice between Grid Search or Randomized Search
        search_type = st.sidebar.radio("Select search type", ('Grid Search', 'Randomized Search'))
        n_iter = 10  # Default number of iterations for Randomized Search
        if search_type == 'Randomized Search':
            n_iter = st.sidebar.slider("Number of Random Combinations to try", 5, 50, 10)

        st.write("Training the Support Vector Regressor...")

        # Base SVR model training for evaluation
        svr_model = SVR(C=C_range[0], kernel=kernel_selected[0], degree=degree_range[0] if kernel_selected[0] == 'poly' else 3, epsilon=epsilon_range[0])
        svr_model.fit(X_train, y_train.values.ravel())

        y_train_pred = svr_model.predict(X_train)
        y_val_pred = svr_model.predict(X_val)
        y_test_pred = svr_model.predict(X_test)

        def evaluate_model(y_true, y_pred, set_name=""):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            correlation = np.corrcoef(y_true.T, y_pred)[0, 1]  # Update to work for multiple target columns
            st.write(f"**{set_name} Set Performance**")
            st.write(f"RMSE: {rmse}")
            st.write(f"MAE: {mae}")
            st.write(f"Correlation: {correlation}")
            st.write("-----")

        evaluate_model(y_train, y_train_pred, "Training")
        evaluate_model(y_val, y_val_pred, "Validation")
        evaluate_model(y_test, y_test_pred, "Test")

        # Parameter grid setup for Grid or Randomized Search
        param_grid = {
            'C': C_range,
            'kernel': kernel_selected,
            'degree': degree_range if 'poly' in kernel_selected else [3],  # Degree for 'poly'
            'epsilon': epsilon_range
        }

        if search_type == 'Grid Search':
            st.write("Performing Grid Search with user-selected hyperparameters...")
            grid_search = GridSearchCV(SVR(), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
            grid_search.fit(X_train, y_train.values.ravel())
            st.write(f"Best Hyperparameters: {grid_search.best_params_}")

            results = pd.DataFrame(grid_search.cv_results_)
        else:
            st.write("Performing Randomized Search with user-selected hyperparameters...")
            random_search = RandomizedSearchCV(SVR(), param_distributions=param_grid, n_iter=n_iter, cv=3, scoring='neg_mean_squared_error', verbose=1)
            random_search.fit(X_train, y_train.values.ravel())
            st.write(f"Best Hyperparameters from Randomized Search: {random_search.best_params_}")

            results = pd.DataFrame(random_search.cv_results_)

        results['mean_test_rmse'] = np.sqrt(-results['mean_test_score'])

        # RMSE Plot
        st.write("RMSE Plot:")
        plt.figure(figsize=(10, 5))
        plt.plot(results.index, results['mean_test_rmse'], label="RMSE")
        plt.xlabel('Grid/Random Search Run')
        plt.ylabel('RMSE')
        plt.legend()
        st.pyplot(plt)

        # Correlation Matrix with Heatmap
        st.write("Correlation Heatmap:")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)

else:
    st.write("Please upload a CSV file to begin.")
