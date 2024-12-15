import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset.csv')

# Display basic information about the dataset
print("Dataset Information:\n")
data.info()
print("\nFirst few rows of the dataset:\n")
print(data.head())

# Inspect features: Analyze each column
categorical_features = []
numerical_features = []

# Classify columns as categorical or numerical
for column in data.columns:
    if data[column].dtype == 'object' or data[column].nunique() <= 10:  # Heuristic for categorical
        categorical_features.append(column)
    else:
        numerical_features.append(column)

print("\nCategorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

# Plot histograms for each feature
print("\nPlotting histograms for all features...")

for column in data.columns:
    if column == "Year":
        plt.figure(figsize=(8, 6))
        
        # Create a countplot for the 'Year' column
        sns.countplot(x=data[column], order=sorted(data[column].unique()), palette='viridis')
        
        plt.title(f"Numerical: {column}")
        plt.ylabel('Frequency')
        plt.xlabel(column)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability (if needed)
        plt.tight_layout()
        plt.savefig(f"charts/{column}.png")
    elif column in numerical_features:
        plt.figure(figsize=(8, 6))
        
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter the data to exclude outliers
        filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)][column]

        sns.histplot(filtered_data, kde=True, color='blue')
        plt.title(f"Numerical: {column}")
        plt.ylabel('Frequency')
        plt.xlabel(column)
        plt.savefig(f"charts/histogram_{column}.png")
        
    

# Check skewness for numerical features
print("\nChecking skewness of numerical features:")
skewness = data[numerical_features].skew()
print(skewness)

# Provide a summary of the distributions
print("\nFeatures Summary:")
for column in numerical_features:
    print(f"{column}: Mean={data[column].mean():.2f}, Median={data[column].median():.2f}, Skewness={skewness[column]:.2f}")
    
# Get top 20 counts for each categorical variable
top_genre = data['genre'].value_counts().head(20)
top_track_names = data['track_name'].value_counts().head(20)
top_track_artists = data['track_artist'].value_counts().head(20)


plt.figure(figsize=(10, 6))
top_genre.plot(kind='bar', color='skyblue')
plt.title('Top 10 Genres')
plt.ylabel('Count')
plt.xlabel('Genre')
plt.tight_layout()
plt.savefig('charts/top_genres.png')

plt.figure(figsize=(10, 6))
top_track_names.plot(kind='bar', color='lightgreen')
plt.title('Top 20 Track Names')
plt.ylabel('Count')
plt.xlabel('Track Name')
plt.tight_layout()
plt.savefig('charts/top_track_names.png')

plt.figure(figsize=(10, 6))
top_track_artists.plot(kind='bar', color='salmon')
plt.title('Top 20 Track Artists')
plt.ylabel('Count')
plt.xlabel('Track Artist')
plt.tight_layout()
plt.savefig('charts/top_track_artists.png')
