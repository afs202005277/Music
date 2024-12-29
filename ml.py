import joblib
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost.sklearn import XGBClassifier  # Use sklearn wrapper for XGBoost

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
    combined = pd.concat([X, y], axis=1)
    augmented = resample(combined, replace=True, n_samples=len(combined) * 2, random_state=42)
    return augmented.drop(columns=["isHit"]), augmented["isHit"]


# Sliding window setup
def sliding_window_split(data, start_year, window_size=5):
    train_data = data[(data['Year'] >= start_year) & (data['Year'] < start_year + window_size)]
    test_data = data[data['Year'] == start_year + window_size]
    return train_data, test_data


def main(window_size, save_df_file):
    data = pd.read_csv("dataset.csv")
    data['artist_id'], artist_mapping = pd.factorize(data['track_artist'])
    data['genre_id'], genre_mapping = pd.factorize(data['genre'])
    data['isHit'] = data['Number of Weeks On Top'] > 0
    data = data.drop(
        columns=["Spotify ID", 'track_name', 'track_artist', 'genre', 'Number of Weeks On Top', 'track_popularity'])
    data = data.dropna()

    reverse_mapping_artist = {index: artist for index, artist in enumerate(artist_mapping)}
    reverse_mapping_genre = {index: genre for index, genre in enumerate(genre_mapping)}

    # Track results
    results = []
    # Perform sliding window approach
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
    data = pd.read_csv("dataset.csv")
    unique_years = sorted(data['Year'].unique())
    max_window_size = len(unique_years) - 1  # Maximum size excluding the last year for testing
    df_results = []
    for window_size in range(5, max_window_size + 1):
        save_df_file = f'results/df_results_window_{window_size}.csv'
        print(f"Running tests with sliding window size: {window_size}")
        df_results += main(window_size, save_df_file)
    pd.DataFrame(df_results).to_csv('unified_df.csv', index=False)


if __name__ == '__main__':
    run()
