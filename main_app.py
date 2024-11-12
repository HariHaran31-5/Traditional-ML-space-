import streamlit as st
import pandas as pd
import subprocess
import os

# Set the page title and layout
st.set_page_config(page_title="Machine Learning Model Portal", layout="wide")

# Main title for the app
st.title("Welcome to the Machine Learning Model Portal")

# Function to handle file upload
def upload_file():
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
    if uploaded_file is not None:
        # Load the CSV into a DataFrame
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())
        # Save the uploaded file temporarily
        file_path = os.path.join("temp_data.csv")
        data.to_csv(file_path, index=False)
        return file_path
    else:
        return None

# Function to run the selected script with the uploaded dataset
def run_script(script_name, data_path):
    if data_path:
        # Run the selected model script
        result = subprocess.run(["streamlit", "run", script_name, "--", data_path], capture_output=True, text=True)
        st.text(result.stdout)
        st.text(result.stderr)
    else:
        st.warning("Please upload a dataset before running the model.")

# Upload dataset
file_path = upload_file()

# List of available model scripts in the 'models/' directory
model_scripts = {
    "Linear Regression": "models/LinearRegression_Regression.py",
    "Random Forest Regression": "models/RandomForest_Regression.py",
    "XGBoost Regression": "models/XGBoost_Regression.py",
    "Random Forest Classification": "models/RandomForest_Classification.py",
    "DB Scan Clustering": "models/DB_Scan_2.py",
    "K means Clustering": "models/K_Means_clustering.py",
    "Hierarchical Clustering": "models/Hierarchical_Clustering.py",
    "SVM Classification": "models/SVM_Classification.py",
    "SVM Regression": "models/SVM_Regression.py"
}

# Drop-down menu for selecting a model
selected_model = st.selectbox("Select a Machine Learning Model", list(model_scripts.keys()))

# Button to run the selected model
if st.button(f"Run"):
    if file_path:
        # Run the script corresponding to the selected model
        run_script(model_scripts[selected_model], file_path)
    else:
        st.warning("Please upload a dataset before running the model.")

# Footer or additional information
st.markdown("---")
st.write("Make sure your dataset is properly formatted before running the model. After selecting a model from the drop-down, click the button to execute it.")





