
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def tune_model():
    df = pd.read_csv("dataset/heart.csv")
    df = df.drop_duplicates()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000))
            ]),
            "params": {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__solver": ["liblinear", "lbfgs"]
            }
        },

        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },

        "KNN": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier())
            ]),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            }
        }
    }

    results = []
    best_model = None
    best_score = 0
    best_name = ""

    for name, details in models.items():
        print("\nTuning", name, "...")

        grid_search = GridSearchCV(
            details["model"],
            details["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        current_model = grid_search.best_estimator_
        y_pred = current_model.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)

        print("\nBest Parameters:")
        print(grid_search.best_params_)

        print("Cross-Validation Accuracy:", grid_search.best_score_)
        print("Testing Accuracy:", test_accuracy)

        results.append({
            "Model": name,
            "Best Parameters": str(grid_search.best_params_),
            "CV Accuracy (%)": round(grid_search.best_score_ * 100, 2),
            "Testing Accuracy (%)": round(test_accuracy * 100, 2)
        })

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = current_model
            best_name = name

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="CV Accuracy (%)",
        ascending=False
    )

    results_df.to_csv("model_results.csv", index=False)

    print("\nFinal Model Comparison:")
    print(results_df.to_string(index=False))

    print("\nBest Model:", best_name)
    print("Best CV Accuracy:", round(best_score * 100, 2), "%")

    model_data = {
        "model": best_model,
        "features": list(X.columns)
    }

    joblib.dump(model_data, "models/heart_disease_model.pkl")

    print("\nModel saved successfully!")
    print("Comparison results saved to model_results.csv")


if __name__ == "__main__":
    tune_model()
