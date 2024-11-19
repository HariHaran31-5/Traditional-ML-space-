import numpy as np
import pandas as pd
import streamlit as st
from pydantic_settings import BaseSettings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score

# Streamlit UI setup
st.title("DBSCAN Clustering")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check for non-numeric data
    if not np.all(df.select_dtypes(include=[np.number]).columns == df.columns):
        st.error("The dataset contains non-numeric columns. Please upload a numeric dataset.")
    else:
        # Show basic info about the dataset
        st.subheader("Data Overview")
        st.write(df.head())
        st.write(df.describe())

        # Heatmap of correlations
        st.subheader("Correlation Heatmap")
        plt.style.use('dark_background')
        plt.figure(figsize=(17, 8))
        sns.heatmap(df.corr(), annot=True, square=True, cmap='tab10')
        st.pyplot(plt.gcf())

        # Pandas profiling report
        st.subheader("Pandas Profiling EDA")
        viz = ProfileReport(df, explorative=True)
        st_profile_report(viz)

        # Sidebar for user input
        st.sidebar.title("DBSCAN Parameters")
        scale_option = st.sidebar.selectbox("Select Scaling Method", ("StandardScaler", "MinMaxScaler"))
        eps = st.sidebar.slider("Epsilon (eps)", min_value=0.01, max_value=1.0, value=0.09, step=0.01)
        min_samples = st.sidebar.slider("Min Samples", min_value=1, max_value=10, value=4, step=1)
        metric_option = st.sidebar.selectbox("Select Distance Metric", 
                                      ("euclidean", "manhattan", "chebyshev", "minkowski"))
    
        leaf_size = st.sidebar.number_input("Leaf Size", min_value=1, max_value=100, value=30)
        p_value = st.sidebar.number_input("Power Parameter (p)", min_value=1, max_value=10, value=2)

        # Scaling the data
        st.subheader("Data Preprocessing")
        if scale_option == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        data = scaler.fit_transform(df)

        st.write("Scaled Data Overview")
        st.write(pd.DataFrame(data, columns=df.columns).describe())

        # Nearest Neighbors plot
        st.subheader("Nearest Neighbors Distance Plot")
        neighbours = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neighbours.fit(data)
        dis, ind = nbrs.kneighbors(data)
        dis = np.sort(dis, axis=0)
        dis = dis[:, -1]

        plt.figure(figsize=(8, 8))
        plt.plot(dis)
        plt.title("Nearest Neighbors Distance Plot")
        plt.xlabel("Points sorted by distance")
        plt.ylabel("Distance")
        st.pyplot(plt.gcf())

        # DBSCAN Model
        st.subheader("DBSCAN Clustering")
        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric=metric_option, leaf_size=leaf_size, p=p_value)
        labels = cluster.fit_predict(data)
        df['Label'] = labels

        st.write("Cluster labels:", df['Label'].unique())

        # Calculate and display clustering evaluation metrics
        if len(set(df['Label'])) > 1:  # More than one cluster
            silhouette_avg = silhouette_score(data, df['Label'])
            db_index = davies_bouldin_score(data, df['Label'])
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")
            st.write(f"Davies-Bouldin Index: {db_index:.2f}")

            # Adjusted Rand Index and Normalized Mutual Information (if true labels are known)
            if 'True Labels' in df.columns:  # Assuming 'True Labels' column exists
                ari = adjusted_rand_score(df['True Labels'], df['Label'])
                nmi = normalized_mutual_info_score(df['True Labels'], df['Label'])
                st.write(f"Adjusted Rand Index: {ari:.2f}")
                st.write(f"Normalized Mutual Information: {nmi:.2f}")

        # Scatter plot of clusters (if data is 2D or can be plotted)
        st.subheader("Cluster Visualization")
        plt.figure(figsize=(15, 12))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=df['Label'], palette='tab10', s=200)
        plt.title("DBSCAN Cluster Visualization")
        st.pyplot(plt.gcf())

        # # Download option for the clustered dataset
        # st.subheader("Download Clustered Dataset")
        # csv = df.to_csv(index=False).encode('utf-8')
        # st.download_button(label="Download CSV", data=csv, file_name='clustered_wine_data.csv', mime='text/csv')
