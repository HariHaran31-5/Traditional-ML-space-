import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
import io  # Import io for string buffer

# Title and description for the UI
st.title('Random Forest Classifier')
st.write("""
### A Random Forest classifier on any dataset you upload
""")

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    dataset = pd.read_csv(uploaded_file)
    
    # Display dataset information
    st.write("### Dataset Preview:")
    st.dataframe(dataset.head())  # Show the first few rows of the dataset

    st.write("### Dataset Information:")
    
    # Capture the info() output
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    info_str = buffer.getvalue()  # Get the string value of the buffer
    st.text(info_str)  # Display dataset information in text format

    st.write("### Dataset Description:")
    st.dataframe(dataset.describe())  # Show descriptive statistics of the dataset

    # Feature and target selection
    X = dataset.iloc[:, [2, 3]].values  # Select features manually, can modify for dynamic selection
    y = dataset.iloc[:, -1].values

    # Splitting dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Sidebar options for user input
    st.sidebar.write("Model Hyperparameters")
    n_estimators = st.sidebar.slider('Number of trees in the forest (n_estimators)', 10, 200, 110)
    max_depth = st.sidebar.slider('Maximum depth of the tree (max_depth)', 1, 20, 5)
    criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
    min_samples_split = st.sidebar.slider('Minimum samples to split a node (min_samples_split)', 2, 10, 2)
    min_samples_leaf = st.sidebar.slider('Minimum samples in a leaf node (min_samples_leaf)', 1, 10, 1)

    # Training the Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                        max_depth=max_depth, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf)
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    # Evaluation Metrics
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bias = classifier.score(X_train, y_train)
    variance = classifier.score(X_test, y_test)

    # Display evaluation metrics
    st.write(f"### Evaluation Metrics")
    st.write(f"Accuracy Score: {ac:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    st.write(f"Model Bias (Training Accuracy): {bias:.2f}")
    st.write(f"Model Variance (Testing Accuracy): {variance:.2f}")

    # Confusion matrix visualization
    st.write("### Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

    # ROC Curve and AUC
    st.write("### ROC Curve and AUC:")
    y_proba = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Feature importance (optional)
    if st.checkbox('Show feature importances'):
        st.write("### Feature Importances:")
        feature_importances = pd.Series(classifier.feature_importances_, index=[f'Feature {i+1}' for i in range(X.shape[1])])
        st.bar_chart(feature_importances)

else:
    st.write("Please upload a CSV file to continue.")