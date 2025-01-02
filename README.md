# Music Data Analysis and Prediction Project
## Project Overview
This project aims to analyze musical data, explore features that contribute to hit songs, and build predictive models to determine whether a track will become a hit. Using datasets from Spotify and Billboard, the goal is to extract actionable insights, generate meaningful visualizations, and develop machine learning models to predict hits. It involves a combination of data extraction, preprocessing, analysis, visualization, and machine learning techniques.
This repository consists of various Python scripts, each handling distinct aspects of the workflow, including data collection, feature engineering, exploratory data analysis, machine learning model development, and visualization of results.
## Script Summary
### 1. **`create_dataset.py`**
- **Purpose**: Handles data collection and preparation.
- **Features**:
    - Fetches songs from Billboard's Hot 100 charts over a specified time period.
    - Retrieves Spotify IDs for the tracks using the Spotify API.
    - Merges and combines datasets from multiple sources to create a unified dataset for analysis.
    - Handles preprocessing steps such as matching tracks, feature normalization, and filtering duplicates.

- **Output**: Outputs a `dataset.csv` file containing all the normalized and merged song information for subsequent processing.

### 2. **`ml.py`**
- **Purpose**: Implements machine learning models for hit song prediction.
- **Features**:
    - Splits the dataset using a sliding window approach.
    - Trains and evaluates multiple models (e.g., RandomForest, XGBoost, GradientBoosting, and KNN) with cross-validation and hyperparameter tuning.
    - Compares scalers (e.g., StandardScaler, MinMaxScaler) for preprocessing.
    - Saves the best-performing models along with their hyperparameters and evaluation metrics.
    - Includes functions for feature importance analysis and plotting.

- **Output**: Trained models saved in the `models/` directory and result summaries in the form of CSV files (main file is `ml_results.csv`).

### 3. **`artist_analysis.py`**
- **Purpose**: Analyzes the contribution of top artists to hit songs using feature comparisons.
- **Features**:
    - Identifies the most influential artists based on the frequency of their hits.
    - Compares key musical features (e.g., energy, danceability) of an artist's tracks against dataset averages.
    - Visualizes variability in musical features for individual artists compared to the dataset average.

- **Output**: Saves artist variability plots and CSV files containing numerical comparisons.

### 4. **`histogram_analysis.py`**
- **Purpose**: Performs in-depth analysis of song attributes using histograms and correlation.
- **Features**:
    - Plots histograms for various numerical attributes (e.g., danceability, loudness) to show their distribution.
    - Analyzes outliers and extracts top deviations from the mean for each feature using Z-scores.
    - Generates a correlation heatmap to understand the relationships between different attributes.

- **Output**: Saves histogram and heatmap visualizations for feature distributions and correlations.

### 5. **`compare_hits.py`**
- **Purpose**: Compares hits based on years and the duration they stayed at the top.
- **Features**:
    - Compares hit song features (e.g., energy, tempo, loudness) across two distinct years.
    - Groups songs into high and low popularity categories based on their number of weeks at the top and compares feature averages.
    - Generates bar charts to visualize attribute differences between years and popularity categories.

- **Output**: Saves comparison plots for feature differences across years and song categories.

### 6. **`data_analysis.py`**
- **Purpose**: Performs exploratory data analysis on the processed dataset.
- **Features**:
    - Categorizes dataset features as numerical or categorical and analyzes distributions.
    - Plots histograms and graphs for numerical data, excluding outliers for clarity.
    - Displays the most popular genres, track names, and artists.
    - Provides summary statistics (e.g., skewness, mean, median) for numerical columns.

- **Output**: Visualizations including histograms for distribution and bar charts for categorical data such as genres and top artists.

### 7. **`features_evolution.ipynb`**
- **Purpose**: Provides a detailed temporal analysis of musical features and trends across years, enriched with artist and genre-specific visualizations, highlighting key shifts in the industry, and analyzing contributions from notable artists and genres.
- **Features**:
    - **Feature Evolution Analysis:**
        - Computes and visualizes the yearly evolution of key musical features such as energy, tempo, loudness, danceability, etc.
        - Generates line plots to display changes in features over time.

    - **Artist and Genre Exploration:**
        - Constructs word clouds for visualizing the most popular artists and genres for specific years (e.g., 2011 and 2016).
        - Provides comparisons between top artists to highlight their contributions and key features.
        - Highlights contributions of top artists, including their evolution and relevance over time.

    - **Insights into Specific Years:**
        - Focuses on specific years (e.g., 2011 and 2016) to analyze energy contributions and musical trends.
        - Highlights top artists through bar charts showing their feature dominance (e.g., peak energy, number of hits).

    - **Artist-Level Analysis:**
        - Tracks the evolution of hits by individual artists over time.
        - Compares artist-specific feature averages against overall dataset averages.

    - **Custom Explorations:**
        - Integrates summary statistics for features by specific artists.
        - Visualizes the effect of individual artist contributions (e.g., Drake) on dataset-wide trends.

- **Output**:
    - Multiple visualizations, including line plots for temporal trends, bar charts for top contributors, and word clouds for genres/artists.
    - In addition to visual output, it provides insights into the evolution and influence of music over time.

### 8. **`musicHitClustering.py`**
- **Purpose**: Leverages clustering algorithms and dimensionality reduction to identify patterns and group similarities within musical features of hit songs.
- **Features**:
    - **Clustering by Feature Groups**:
        - Divides features into distinct groups for focused clustering:
            1. **Energy-Based** (e.g., tempo, loudness).
            2. **Acoustic-Based** (e.g., valence, acousticness).
            3. **Danceability-Based** (e.g., danceability, speechiness, duration).

        - Executes clustering separately for each group to identify meaningful patterns in each domain.

    - **Preprocessing**:
        - Handles missing data by imputing mean values for features.
        - Standardizes feature scales using `StandardScaler` to ensure valid clustering results.

    - **K-Means Clustering**:
        - Utilizes the K-Means algorithm to group songs based on specific feature sets.
        - Employs the **Elbow Method** to determine the optimal number of clusters, minimizing inertia while avoiding overfitting.

    - **Dimensionality Reduction**:
        - Applies Principal Component Analysis (PCA) for feature reduction, projecting data into 2D space.
        - Explains variance percentages contributed by the PCA components.

    - **Visualization**:
        - Generates scatter plots for clusters in 2D space.
        - Annotates visual clusters with axis labels based on dominant features contributing to PCA components.
        - Visualizes the Elbow Method curves for clarity in selecting the optimal number of clusters.

    - **Feature Contributions**:
        - Analyzes and displays feature contributions for each PCA dimension.
        - Highlights the most influential features for cluster separation.

- **Output**:
    - Interactive visualizations for each feature cluster:
        - **Elbow Method Plot**: To identify the optimal number of clusters.
        - **PCA Scatter Plot**: Showcases clusters in 2D space with feature-based axis labels.

    - Feature contribution tables for each PCA component, aiding interpretability.

## Folder Structure
- **`datasets/`**: Contains raw CSV files for input data from various sources.
- **`models/`**: Stores trained machine learning models.
- **`results/`**: Holds CSV files summarizing the model results.
- **`charts/`**: Directory for saving visualizations like histograms and bar charts.
- **`artist_analysis/`**: Contains artist-specific plots and analysis files.
- **`feature_importances/`**: Saves feature importance plots and related data.

## Important Notice
To run the dataset creation, the necessary datasets need to be previously downloaded into the `datasets/` folder, with appropriate names:

| Dataset URL                                                    | Dataset Name         |
|---------------------------------------------------------------|----------------------|
| [https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs](https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs) | 114000.csv          |
| [https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019) | 2000_2019.csv       |
| [https://www.kaggle.com/datasets/muhmores/spotify-top-100-songs-of-20152019](https://www.kaggle.com/datasets/muhmores/spotify-top-100-songs-of-20152019) | 2010_2019.csv       |
| [https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks) | spotify_data.csv    |
| [https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) | 30000.csv           |


## Authors (Group 7)
| Name            | Number      |
|-----------------|-------------|
| Alexandre Nunes | up202005358 |
| André Sousa     | up202005277 |
| Gonçalo Pinto   | up202004907 |
| Maria Gonçalves | up202006927 |
