import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import os

# Load the dataset
data = pd.read_csv('dataset.csv')
data = data[data['Number of Weeks On Top'] > 0]

# Drop unnecessary columns
data.drop(['genre', 'Spotify ID'], axis=1, inplace=True)

numerical_features = data.columns.drop(['track_name', 'track_artist', 'Year'])

print("\nPlotting correlation heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("hist_analysis/correlation_heatmap.png")
plt.show()

# Analyze histograms and find top 5 deviations
results = []

for feature in numerical_features:
    print(f"\nAnalyzing feature: {feature}")
    
    print(data[feature].mean())
    
    # Calculate Z-scores for the feature
    data[f'zscore_{feature}'] = abs(zscore(data[feature]))
    
    # Plot histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True, color='blue')
    plt.title(f"Histogram: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.savefig(f"hist_analysis/histogram_{feature}.png")
    plt.close()
    
    top_deviations = data.nlargest(10, f'zscore_{feature}')[['Year', 'track_name', 'track_artist', feature, f'zscore_{feature}']]
    results.append({
        'feature': feature,
        'top_songs': top_deviations
    })

    # Print the results for this feature
    print(f"Top 10 deviations for {feature}:")
    print(top_deviations)