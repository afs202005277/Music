import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset.csv')

# Filter only the hits (songs with Number of Weeks On Top > 0)
data = data[data['Number of Weeks On Top'] > 0]

# Define feature groups for clustering
feature_groups = {
    'Energy-Based': ['energy', 'loudness', 'tempo'],
    'Acoustic-Based': ['acousticness', 'instrumentalness', 'valence'],
    'Danceability-Based': ['danceability', 'speechiness', 'duration_ms']
}

# Process each feature group
cluster_descriptions = {}

for group_name, features in feature_groups.items():
    print(f'Clustering for {group_name} features: {features}')

    # Handle missing values and scaling
    data[features] = data[features].fillna(data[features].mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    # Calculate elbow curve
    inertia = []
    range_clusters = range(1, 11)
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range_clusters, inertia, marker='o')
    plt.title(f'Elbow Method for Optimal k - {group_name}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Fit PCA and get component information
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_ * 100

    # Get feature contributions to each component
    feature_weights = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=features
    )

    # Determine dominant features for each component
    pc1_features = feature_weights['PC1'].abs().nlargest(2)
    pc2_features = feature_weights['PC2'].abs().nlargest(2)

    # Create meaningful axis labels
    pc1_label = f"PC1 ({explained_variance[0]:.1f}%): " + " & ".join(pc1_features.index)
    pc2_label = f"PC2 ({explained_variance[1]:.1f}%): " + " & ".join(pc2_features.index)

    # Fit KMeans and get cluster labels
    optimal_k = 4  # Replace with your optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot clusters
    for cluster in range(optimal_k):
        cluster_points = pca_data[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=f'Cluster {cluster}', alpha=0.6)

    # Add annotations
    plt.title(f'{group_name} Feature Clusters\nTotal Explained Variance: {sum(explained_variance):.1f}%',
              pad=20)
    plt.xlabel(pc1_label)
    plt.ylabel(pc2_label)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Print feature contributions
    print(f"\nFeature contributions to principal components for {group_name}:")
    print(feature_weights)

    plt.tight_layout()
    plt.show()
