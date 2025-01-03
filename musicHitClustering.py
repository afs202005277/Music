import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = pd.read_csv('dataset.csv')


data = data[data['Number of Weeks On Top'] > 0]


feature_groups = {
    'Energy-Based': ['energy', 'loudness', 'tempo'],
    'Acoustic-Based': ['acousticness', 'instrumentalness', 'valence'],
    'Danceability-Based': ['danceability', 'speechiness', 'duration_ms']
}


cluster_descriptions = {}

for group_name, features in feature_groups.items():
    print(f'Clustering for {group_name} features: {features}')


    data[features] = data[features].fillna(data[features].mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])


    inertia = []
    range_clusters = range(1, 11)
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)


    plt.figure(figsize=(8, 5))
    plt.plot(range_clusters, inertia, marker='o')
    plt.title(f'Elbow Method for Optimal k - {group_name}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()


    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)


    explained_variance = pca.explained_variance_ratio_ * 100


    feature_weights = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=features
    )


    pc1_features = feature_weights['PC1'].abs().nlargest(2)
    pc2_features = feature_weights['PC2'].abs().nlargest(2)


    pc1_label = f"PC1 ({explained_variance[0]:.1f}%): " + " & ".join(pc1_features.index)
    pc2_label = f"PC2 ({explained_variance[1]:.1f}%): " + " & ".join(pc2_features.index)


    optimal_k = 4  
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)


    plt.figure(figsize=(12, 8))


    for cluster in range(optimal_k):
        cluster_points = pca_data[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=f'Cluster {cluster}', alpha=0.6)


    plt.title(f'{group_name} Feature Clusters\nTotal Explained Variance: {sum(explained_variance):.1f}%',
              pad=20)
    plt.xlabel(pc1_label)
    plt.ylabel(pc2_label)


    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


    print(f"\nFeature contributions to principal components for {group_name}:")
    print(feature_weights)

    plt.tight_layout()
    plt.show()
