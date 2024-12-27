import json

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

models = {
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
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=500, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"]
        }
    },
    "SVM": {
        "model": SVC(random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
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


def augment_data(X, y):
    combined = pd.concat([X, y], axis=1)
    augmented = resample(combined, replace=True, n_samples=len(combined) * 2, random_state=42)
    return augmented.drop(columns=["isHit"]), augmented["isHit"]


# Sliding window setup
def sliding_window_split(data, start_year, window_size=5):
    train_data = data[(data['Year'] >= start_year) & (data['Year'] < start_year + window_size)]
    test_data = data[data['Year'] == start_year + window_size]
    return train_data, test_data


def main(window_size):
    data = pd.read_csv("dataset.csv")
    data['isHit'] = data['Number of Weeks On Top'] > 0
    data = data.drop(columns=["Spotify ID"])
    data = data.dropna()

    # Track results
    results = []

    # Perform sliding window approach
    unique_years = sorted(data['Year'].unique())

    for start_year in unique_years[:-window_size]:
        train_data, test_data = sliding_window_split(data, start_year, window_size)

        if train_data.empty or test_data.empty:
            continue

        X_train = train_data.drop(columns=["isHit", "Year"])
        y_train = train_data["isHit"]
        X_test = test_data.drop(columns=["isHit", "Year"])
        y_test = test_data["isHit"]

        X_train, y_train = augment_data(X_train, y_train)

        for model_name, model_info in models.items():
            grid = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            predictions = best_model.predict(X_test)

            acc = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            results.append({
                "Start Year": start_year,
                "Model": model_name,
                "Best Params": grid.best_params_,
                "Accuracy": acc,
                "Classification Report": report
            })
            print(f"Finished model {model_name} in year {start_year}")

        for result in results:
            print(f"Sliding Window Start Year: {result['Start Year']}")
            print(f"Model: {result['Model']}")
            print(f"Best Params: {result['Best Params']}")
            print(f"Accuracy: {result['Accuracy']:.2f}")
            print(f"Classification Report:\n{result['Classification Report']}\n")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main(5)
