import numpy as np
import pandas as pd
import streamlit as st
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import silhouette_score

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


st.title("K-Means Clustering")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check for non-numeric data
    if not np.all(df.select_dtypes(include=[np.number]).columns == df.columns):
        st.error("The dataset contains non-numeric columns. Please upload a numeric dataset.")
    else:



        # Show basic info about the dataset
        st.write("Dataset Information:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s) 

        st.write("Statistical Summary:")
        st.write(df.describe())

        st.write("Dataset Preview:")
        st.write(df.head())

        # Pandas profiling report
        st.subheader("Pandas Profiling EDA")
        viz = ProfileReport(df, explorative=True)
        st_profile_report(viz)

        # Heatmap of correlations
        st.subheader("Correlation Heatmap")
        plt.style.use('dark_background')
        plt.figure(figsize=(17, 8))
        sns.heatmap(df.corr(), annot=True, square=True, cmap='tab10')
        st.pyplot(plt.gcf())

        # Sidebar for user input
        st.sidebar.title("K-Means Parameters")
        scale_option = st.sidebar.selectbox("Select Scaling Method", ("StandardScaler", "MinMaxScaler"))
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3, step=1)
        max_iter = st.sidebar.slider("Max Iterations", min_value=100, max_value=1000, value=300, step=100)
        random_state = st.sidebar.number_input("Random State", min_value=0, max_value=100, value=42)

        # Scaling the data
        st.subheader("Data Preprocessing")
        if scale_option == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        data = scaler.fit_transform(df)

        st.write("Scaled Data Overview")
        st.write(pd.DataFrame(data, columns=df.columns).describe())

        #User-Defined Feature Selection
        selected_features = st.multiselect("Select features for clustering", df.columns)
        if len(selected_features) > 1:
            data = df[selected_features]
            data = scaler.fit_transform(data)
        else:
            st.warning("Please select at least two features.")


        # K-Means Model
        st.subheader("K-Means Clustering")
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        labels = kmeans.fit_predict(data)
        df['Label'] = labels

        st.write("Cluster labels:", df['Label'].unique())


        # Elbow Method
        st.subheader("Elbow Method for Optimal Clusters")
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        st.pyplot(plt.gcf())

        # Silhouette Analysis
        st.subheader("Silhouette Analysis for Optimal Clusters")
        silhouette_avg_list = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_avg_list.append(silhouette_avg)

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 11), silhouette_avg_list, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Average Silhouette Score')
        plt.title('Silhouette Analysis showing optimal number of clusters')
        st.pyplot(plt.gcf())


        # #Cluster Center Visualization
        # st.subheader("Cluster Centers")
        # st.subheader("Display the cluster centers for users to visualize where the centroids of clusters are located.")
        # centers = kmeans.cluster_centers_
        # st.write(pd.DataFrame(centers, columns=df.columns))


        #Outlier Detection
        st.subheader("Outlier Detection (Z-score Method)")
        z_scores = np.abs(stats.zscore(df))
        outliers = (z_scores > 3).any(axis=1)
        st.write("Number of outliers:", np.sum(outliers))

        
        #cluster summary
        st.subheader("Cluster Summary")
        cluster_summary = df.groupby('Label').mean()
        st.write(cluster_summary)


        # Scatter plot of clusters (if data is 2D or can be plotted)
        st.subheader("Cluster Visualization")
        plt.figure(figsize=(15, 12))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=df['Label'], palette='tab10', s=200)
        plt.title("K-Means Cluster Visualization")
        st.pyplot(plt.gcf())

        #3D Scatter Plot for Clustering
        st.subheader("3D Cluster Visualization")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=df['Label'], cmap='tab10', s=200)
        plt.title("3D K-Means Cluster Visualization")
        st.pyplot(plt.gcf())






        
        # # Calculate and display clustering evaluation metrics
        # silhouette_avg = silhouette_score(data, df['Label'])
        # db_index = davies_bouldin_score(data, df['Label'])
        # st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        # st.write(f"Davies-Bouldin Index: {db_index:.2f}")



