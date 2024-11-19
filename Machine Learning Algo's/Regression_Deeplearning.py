import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import io

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  # Forward-fill missing values
    df.drop_duplicates(inplace=True)
    
    # Convert data types (example)
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype('int')
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)  # Use drop_first to avoid dummy variable trap

    return df

st.title("Deep Neural Network Regression")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file, encoding='utf-8')

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



    # Split and scale data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define model creation function
    def create_model(learning_rate=0.001):
        model = Sequential([
            Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Linear activation for regression
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # Sidebar for hyperparameter tuning
    st.sidebar.header("Hyperparameter Tuning")
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.0005, 0.0001])
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128])
    epochs = st.sidebar.selectbox("Epochs", [10, 20, 30])

    # Train the model
    st.write(f"Training with learning rate={learning_rate}, batch size={batch_size}, epochs={epochs}")
    model = create_model(learning_rate=learning_rate)
    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val_scaled, y_val), verbose=0)

    # Predict and evaluate
    y_test_pred = model.predict(X_test_scaled).ravel()
    mse = mean_squared_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)

    st.write(f"**Test Set Performance**")
    st.write(f"MSE: {mse}")
    st.write(f"MAPE: {mape}")

    # Visualize training history
    st.write("Training History:")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(plt)

    # Show the performance metrics in a table format
    results_df = pd.DataFrame([{
        "Learning Rate": learning_rate, 
        "Batch Size": batch_size, 
        "Epochs": epochs, 
        "MSE": mse, 
        "MAPE": mape
    }])
    st.write("Performance Metrics:")
    st.write(results_df)
else:
    st.write("Please upload a CSV file to begin.")
