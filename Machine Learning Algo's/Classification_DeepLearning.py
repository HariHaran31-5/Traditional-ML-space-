import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
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

st.title("Deep Learning Classifier")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='utf-8')

    st.write("Dataset Information:")
    st.write(data.info())
    st.write("Dataset Preview:")
    st.write(data.head())

    data = preprocess_data(data)
    st.write("Preprocessed Dataset:")
    st.write(data.head())

    # Select target and feature variables
    st.write("Select Feature and Target Variables:")
    target_columns = st.multiselect("Select target variable(s)", data.columns)
    feature_columns = st.multiselect("Select feature variable(s)", data.columns)

    if not target_columns or not feature_columns:
        st.warning("Please select both features and target variables.")
    else:
        X = data[feature_columns]
        y = data[target_columns]

        # Handle target encoding for multi-class classification
        if y.shape[1] == 1 and y.dtypes[0] == 'object':
            y = pd.get_dummies(y)

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Convert to NumPy arrays
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        # Handle target encoding for multi-class
        if y_train.ndim == 1 or y_train.shape[1] == 1:
            y_train = to_categorical(y_train)
            y_val = to_categorical(y_val)
            y_test = to_categorical(y_test)

        # Build Keras model
        st.sidebar.header("Model Hyperparameters")
        input_dim = X_train.shape[1]
        num_classes = y_train.shape[1] if len(y_train.shape) > 1 else 1
        hidden_units = st.sidebar.slider("Hidden Units", 16, 256, 64)
        dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
        activation = st.sidebar.selectbox("Activation Function", ['relu', 'tanh', 'sigmoid'], index=0)
        epochs = st.sidebar.slider("Epochs", 1, 100, 20)
        batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

        model = Sequential()
        model.add(Dense(hidden_units, input_dim=input_dim, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_units // 2, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid'))

        model.compile(optimizer='adam', loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        st.write("Training the Deep Learning Model...")
        history = model.fit(
            X_train, 
            y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=1
        )

        # Plot training history
        st.write("Training History:")
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(history.history['accuracy'], label='Training Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].legend()
        ax[0].set_title('Accuracy')

        ax[1].plot(history.history['loss'], label='Training Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].legend()
        ax[1].set_title('Loss')

        st.pyplot(fig)

        # Evaluate the model
        st.write("Evaluating the Model...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1) if num_classes > 1 else (y_pred > 0.5).astype(int)
        y_test_classes = np.argmax(y_test, axis=1) if num_classes > 1 else y_test.ravel()

        st.write("Classification Report:")
        st.text(classification_report(y_test_classes, y_pred_classes))

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm")
        st.pyplot(plt)

else:
    st.write("Please upload a CSV file to begin.")
