import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)  
    df.drop_duplicates(inplace=True)
    
    # Normalize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_scaled

st.title("Hierarchical Clustering")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Information:")
    st.write(data.describe())
    st.write("Dataset Preview:")
    st.write(data.head())

    # Preprocess the data (handle missing values, scaling, etc.)
    data_scaled = preprocess_data(data)
    st.write("Data after Preprocessing (Standard Scaling applied):")
    st.write(data_scaled.head())

    # Select variables for clustering
    st.write("Select Variables for Clustering:")
    columns_for_clustering = st.multiselect("Select feature variable(s)", data_scaled.columns, default=data_scaled.columns)

    if not columns_for_clustering:
        st.warning("Please select at least one variable for clustering.")
    else:
        X = data_scaled[columns_for_clustering]

        # Distance Metrics Selection
        st.sidebar.header("Hierarchical Clustering Parameters")
        distance_metric = st.sidebar.selectbox("Select Distance Metric", ["euclidean", "cityblock", "cosine", "correlation"], index=0)
        linkage_method = st.sidebar.selectbox("Select Linkage Method", ["ward", "single", "complete", "average"], index=0)
        num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)

        # Hierarchical Clustering using Agglomerative Clustering
        st.write(f"Performing Hierarchical Clustering with {linkage_method} linkage and {distance_metric} distance...")
        agglomerative_clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity=distance_metric, linkage=linkage_method)
        cluster_labels = agglomerative_clustering.fit_predict(X)

        # Display cluster labels
        st.write("Cluster Labels for Each Data Point:")
        data_scaled['Cluster_Label'] = cluster_labels
        st.write(data_scaled)

        # Plot Dendrogram using SciPy
        st.write("Dendrogram:")
        linked = linkage(X, method=linkage_method, metric=distance_metric)
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', labels=data.index, distance_sort='descending', show_leaf_counts=True)
        plt.title("Dendrogram")
        st.pyplot(plt)

        # Plot Cluster Heatmap
        st.write("Cluster Heatmap:")
        sns.clustermap(X, method=linkage_method, metric=distance_metric, cmap="coolwarm", standard_scale=1)
        st.pyplot(plt)

        # Cluster Visualization (Scatter Plot)
        if len(columns_for_clustering) >= 2:
            st.write("Cluster Visualization:")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=cluster_labels, palette="coolwarm")
            plt.xlabel(columns_for_clustering[0])
            plt.ylabel(columns_for_clustering[1])
            plt.title("Hierarchical Clustering (First Two Features)")
            st.pyplot(plt)

else:
    st.write("Please upload a CSV file to begin.")
