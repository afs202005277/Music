import joblib
import time

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost.sklearn import XGBClassifier  # Use sklearn wrapper for XGBoost

font = {'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

models = {
    "XGBoost": {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        }
    },
    'KNN': {
        "model": KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
        }
    },
    'DT': {
        "model": DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 7, 9],
            'min_samples_split': [2, 5, 10]
        }
    }
}

scalers = {
    "NoScaling": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler()
}


def augment_data(X, y):
    """
    Performs data augmentation by oversampling the input dataset. Concatenates the feature set and labels, then
    resamples the combined dataset to double its original size, ensuring balanced representation. Returns the
    augmented feature set and labels separately.

    :param X: Input feature set (pandas DataFrame) to augment.
    :param y: Labels (pandas Series) corresponding to the feature set.
    :return: Tuple containing the augmented feature set (pandas DataFrame) without the "isHit" column, and the
        augmented labels (pandas Series).
    """
    combined = pd.concat([X, y], axis=1)
    augmented = resample(combined, replace=True, n_samples=len(combined) * 2, random_state=42)
    return augmented.drop(columns=["isHit"]), augmented["isHit"]


def sliding_window_split(data, start_year, window_size=5):
    """
    Splits a dataset into training and testing sets using a sliding window approach based on
    the year column. The function segregates the data into training and testing sets where
    the training data includes a specified range of years starting from a given year, and
    the testing data includes the following year after the specified range.

    :param data: The input dataset to be processed, typically in the form of a DataFrame,
        which should include a 'Year' column for the sliding window operation.
    :type data: pandas.DataFrame
    :param start_year: The starting point of the sliding window, indicating the first
        year of the training data range.
    :type start_year: int
    :param window_size: The size of the sliding window that determines how many years
        are included in the training data range. Default value is 5.
    :type window_size: int
    :return: A tuple containing two subsets of the input dataset:
        - The first subset ('train_data') includes rows where the 'Year' column falls
          between `start_year` and (`start_year` + `window_size`).
        - The second subset ('test_data') includes rows where the 'Year' column equals
          (`start_year` + `window_size`).
    :rtype: tuple (pandas.DataFrame, pandas.DataFrame)
    """
    train_data = data[(data['Year'] >= start_year) & (data['Year'] < start_year + window_size)]
    test_data = data[data['Year'] == start_year + window_size]
    return train_data, test_data


def main(window_size, save_df_file):
    """
    Executes a data processing and machine learning pipeline on a music dataset to predict whether a track is a hit.
    The function preprocesses the dataset, splits it into training and testing sets using a sliding window approach,
    trains multiple models with different scalers and hyperparameter configurations, evaluates the models, and stores
    the results. The best-performing models are saved for future use.

    :param window_size: The size of the sliding window to split the data temporally, as an integer.
    :param save_df_file: The file path where the results dataframe should be saved, as a string.
    :return: A list of dictionaries containing the results of the model evaluations for each combination of scaler,
        model, and hyperparameter configurations.
    """
    data = pd.read_csv("dataset.csv")
    data['artist_id'], artist_mapping = pd.factorize(data['track_artist'])
    data['genre_id'], genre_mapping = pd.factorize(data['genre'])
    data['isHit'] = data['Number of Weeks On Top'] > 0
    data = data.drop(
        columns=["Spotify ID", 'track_name', 'track_artist', 'genre', 'Number of Weeks On Top', 'track_popularity'])
    data = data.dropna()

    # reverse_mapping_artist = {index: artist for index, artist in enumerate(artist_mapping)}
    # reverse_mapping_genre = {index: genre for index, genre in enumerate(genre_mapping)}

    results = []
    unique_years = sorted(data['Year'].unique())

    for start_year in unique_years[:-window_size]:
        train_data, test_data = sliding_window_split(data, start_year, window_size)

        if train_data.empty or test_data.empty:
            continue

        print(
            f"Training with years {sorted(train_data['Year'].unique())} and testing with year {sorted(test_data['Year'].unique())}")
        X_train = train_data.drop(columns=["isHit", "Year"])
        y_train = train_data["isHit"]
        X_test = test_data.drop(columns=["isHit", "Year"])
        y_test = test_data["isHit"]

        X_train, y_train = augment_data(X_train, y_train)

        for scaler_name, scaler in scalers.items():
            if scaler is None:
                X_train_scaled = X_train
                X_test_scaled = X_test
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            for model_name, model_info in models.items():
                grid = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train_scaled, y_train)

                best_model = grid.best_estimator_
                predictions = best_model.predict(X_test_scaled)

                acc = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                f1 = f1_score(y_test, predictions, zero_division=0)
                report = classification_report(y_test, predictions)
                conf_matrix = confusion_matrix(y_test, predictions)

                joblib.dump(best_model, f'models/{model_name}_{scaler_name}_{start_year}_{window_size}.joblib')

                results.append({
                    "Start Year": start_year,
                    "Window Size": window_size,
                    "Scaler": scaler_name,
                    "Model": model_name,
                    "Best Params": grid.best_params_,
                    "Accuracy": acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Classification Report": str(report),
                    'Confusion Matrix': str(conf_matrix)
                })
                print(
                    f"{time.strftime('%H:%M:%S', time.localtime())}: Finished model {model_name} with scaler {scaler_name} in year {start_year} with window size of {window_size}.")

    results_dataframe = pd.DataFrame(results)
    results_dataframe.to_csv(save_df_file, index=False)

    return results


def run():
    """
    Executes the sliding window test on dataset years using specified window sizes and
    saves the results to CSV files.

    This function performs tests across a range of window sizes, iterating through
    various configurations to produce results. Data is read from a CSV file containing
    a "Year" column, and unique years are used for determining the range of test
    parameters. Results are aggregated and saved to an output CSV.
    :param None: This function does not accept parameters.

    :return: None
    """
    data = pd.read_csv("dataset.csv")
    unique_years = sorted(data['Year'].unique())
    max_window_size = len(unique_years) - 1  # Maximum size excluding the last year for testing
    df_results = []
    for window_size in range(5, max_window_size + 1):
        save_df_file = f'results/df_results_window_{window_size}.csv'
        print(f"Running tests with sliding window size: {window_size}")
        df_results += main(window_size, save_df_file)
    pd.DataFrame(df_results).to_csv('ml_results.csv', index=False)


def get_feature_importances(unified_df, feature_names):
    """
    Analyzes feature importances based on the best-performing RandomForest models from the input DataFrame.

    The function identifies the best RandomForest model for each unique combination of the `Start Year`
    and `Window Size` from the `unified_df` DataFrame. It loads the corresponding models from
    predefined file paths and extracts their feature importances. The resulting feature
    importances are returned as a new DataFrame.

    :param unified_df: The input DataFrame containing model performance data for various combinations of
        parameters. It must contain the following columns: 'Accuracy', 'Model', 'Scaler', 'Start Year',
        and 'Window Size'.
    :param feature_names: A list of feature names used during training of the models, corresponding
        to the order of feature importances.
    :return: A DataFrame containing the calculated feature importances for each model, grouped by
        unique `Start Year` and `Window Size`. If a model with necessary attributes cannot be found,
        its feature importances will not be included in the final output.
    """
    unified_df = unified_df.sort_values(by='Accuracy', ascending=False)
    rf_df = unified_df[unified_df['Model'] == 'RandomForest']

    combinations = rf_df[['Start Year', 'Window Size']].drop_duplicates()

    feature_importances = []

    for _, row in combinations.iterrows():
        start_year = row['Start Year']
        window_size = row['Window Size']

        subset = rf_df[(rf_df['Start Year'] == start_year) &
                       (rf_df['Window Size'] == window_size)]

        best_model_row = subset.iloc[0]
        model_name = best_model_row['Model']
        scaler_name = best_model_row['Scaler']

        model_file_path = f"models/{model_name}_{scaler_name}_{start_year}_{window_size}.joblib"

        try:
            model = joblib.load(model_file_path)

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict = {
                    'Start Year': start_year,
                    'Window Size': window_size
                }
                importance_dict.update({feature: importance for feature, importance in zip(feature_names, importances)})
                feature_importances.append(importance_dict)
            else:
                print(f"Model at {model_file_path} does not have feature_importances_ attribute.")
        except Exception as e:
            print(f"Failed to load model from {model_file_path}: {e}")

    feature_importances_df = pd.DataFrame(feature_importances)
    return feature_importances_df


def plot_average_feature_importances(feature_importances_df, feature_names):
    """
    Plots the average feature importances for the given features and saves the
    plot as an image.

    This function takes an input DataFrame containing feature importances and
    calculates the average importance of specific features. The average
    importances are then visualized as a bar chart, which is saved as a PNG
    file and displayed.

    :param feature_importances_df: DataFrame containing feature importances for
        several features across multiple models or samples.
    :type feature_importances_df: pandas.DataFrame
    :param feature_names: List of feature names for which the average
        importances will be calculated and plotted.
    :type feature_names: list of str
    :return: None
    """
    feature_columns = [col for col in feature_importances_df.columns if col in feature_names]
    averages = feature_importances_df[feature_columns].mean()

    plt.figure(figsize=(10, 6))
    averages.plot(kind='bar', color='skyblue')
    plt.title('Average Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances/average_feature_importances.png')
    plt.show()


def plot_feature_importance_variation(feature_importances_df, feature_names, window_sizes):
    """
    Plots the variation in feature importance over years for given window sizes. The function
    filters the input DataFrame to only include rows corresponding to the specified window sizes,
    and then plots the feature importance values across years for each feature and window size.

    :param feature_importances_df:
        DataFrame containing feature importance values with columns including 'Window Size',
        'Start Year', and feature names. DataFrame must also include numerical 'Start Year'
        for proper sorting.
    :param feature_names:
        List of feature names to be plotted. Each feature corresponds to a column in the
        input DataFrame.
    :param window_sizes:
        List of window sizes to filter the DataFrame. Feature importance will be plotted
        for each window size in the list.
    :return:
        None. The function saves the plots as PNG files and optionally displays them.
    """
    filtered_df = feature_importances_df[feature_importances_df['Window Size'].isin(window_sizes)]

    for window_size in window_sizes:
        subset = filtered_df[filtered_df['Window Size'] == window_size]

        subset = subset[subset['Start Year'] == subset['Start Year'].astype(int)]
        subset = subset.sort_values(by='Start Year')

        plt.figure(figsize=(12, 8))
        for feature in feature_names:
            plt.plot(subset['Start Year'], subset[feature], label=f"{feature}")

        plt.title(f'Feature Importance Variation Over Years (Window Size={window_size})')
        plt.xlabel('Start Year')
        plt.ylabel('Feature Importance')
        plt.legend(loc='right')
        plt.tight_layout()
        plt.savefig(f'feature_importances/evolution_{window_size}.png')
        plt.show()


if __name__ == '__main__':
    run()
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                     'instrumentalness', 'valence', 'tempo', 'duration_ms', 'artist_id',
                     'genre_id']
    feature_importances = get_feature_importances(pd.read_csv("ml_results.csv"),
                                                  feature_names)
    feature_importances.to_csv('feature_importances/feature_importances.csv', index=False)
    plot_average_feature_importances(feature_importances, feature_names)
    plot_feature_importance_variation(feature_importances, feature_names, [5, 10, 15])
