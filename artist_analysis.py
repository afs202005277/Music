import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

font = {'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

def load_and_filter_data(filepath):
    """
    Load the dataset and filter rows where 'Number of Weeks On Top' is greater than 0.
    Args:
        filepath (str): Path to the dataset CSV file.
    Returns:
        pd.DataFrame: Filtered dataset.
    """
    data = pd.read_csv(filepath)
    return data[data['Number of Weeks On Top'] > 0]

def get_top_artists(data, top_n=3):
    """
    Identify the top N most popular artists based on the frequency of their appearance in the dataset.
    Args:
        data (pd.DataFrame): The dataset.
        top_n (int): Number of top artists to identify.
    Returns:
        list: List of top N artist names.
    """
    return data['track_artist'].value_counts().head(top_n).index.tolist()

def compute_artist_and_overall_averages(data, artists):
    """
    Compute average numeric features for each specified artist and overall dataset.
    Args:
        data (pd.DataFrame): The dataset.
        artists (list): List of artist names.
    Returns:
        pd.DataFrame: DataFrame containing artist averages and overall averages for comparison.
    """
    artist_averages = {}
    for artist in artists:
        artist_data = data[data['track_artist'] == artist]
        artist_averages[artist] = artist_data.mean(numeric_only=True)

    overall_average = data.mean(numeric_only=True)

    artist_avg_df = pd.DataFrame(artist_averages).T
    artist_avg_df.index.name = 'Artist'

    overall_avg_df = pd.DataFrame(overall_average).T
    overall_avg_df.index = ['Reference']

    return pd.concat([artist_avg_df, overall_avg_df]).round(3)

def save_comparison_to_csv(comparison_df, output_path):
    """
    Save the comparison DataFrame to a CSV file.
    Args:
        comparison_df (pd.DataFrame): The comparison DataFrame.
        output_path (str): Path to save the CSV file.
    """
    comparison_df.to_csv(output_path)

def normalize_features(data, features):
    """
    Normalize specified features in the dataset using MinMaxScaler.
    Args:
        data (pd.DataFrame): The dataset.
        features (list): List of feature column names to normalize.
    Returns:
        tuple: Normalized dataset and scaler instance.
    """
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    return data, scaler

def plot_artist_variability(data, artists, features, reference_values, output_folder):
    """
    Generate and save boxplots for feature variability of each artist with reference values.
    Args:
        data (pd.DataFrame): The dataset.
        artists (list): List of artist names to analyze.
        features (list): List of features to plot.
        reference_values (list): Normalized reference values for the features.
        output_folder (str): Folder to save the plots.
    """
    for artist in artists:
        artist_data = data[data['track_artist'] == artist]

        # Melt the data for easier plotting
        melted_data = artist_data.melt(id_vars=['track_name'],
                                       value_vars=features,
                                       var_name='Feature', value_name='Value')

        # Create a boxplot
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=melted_data, x='Feature', y='Value')

        # Add normalized reference values as horizontal lines
        for i, feature in enumerate(features):
            ref_value = reference_values[i]
            x_position = features.index(feature)
            plt.plot([x_position - 0.4, x_position + 0.4], [ref_value, ref_value], color='red', linestyle='--')

        plt.plot([], [], color='red', linestyle='--', label='Dataset Average')
        plt.legend(loc='upper right')

        plt.title(f'Feature Variability for {artist}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save each plot
        plt.savefig(output_folder + f"{artist.replace(' ', '_')}_variability_plot.png")
        plt.show()

def main():
    """
    Main function to orchestrate the analysis and visualization of artist data.
    """
    dataset_path = 'dataset.csv'
    output_folder = 'artist_analysis/'
    features_to_normalize = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                             'instrumentalness', 'valence', 'tempo', 'duration_ms']

    # Load and process data
    data = load_and_filter_data(dataset_path)
    top_artists = get_top_artists(data)

    # Compute averages and save comparison
    comparison_df = compute_artist_and_overall_averages(data, top_artists)
    save_comparison_to_csv(comparison_df, output_folder + 'artist_comparison.csv')

    # Normalize features
    data, scaler = normalize_features(data, features_to_normalize)
    normalized_reference = scaler.transform(comparison_df.loc['Reference', features_to_normalize].values.reshape(1, -1))[0]

    # Plot variability
    plot_artist_variability(data, top_artists, features_to_normalize, normalized_reference, output_folder)

    print("Analysis complete. Plots and table saved in the \"artist_analysis\" folder.")

if __name__ == "__main__":
    main()
