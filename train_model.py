
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


def visualize_model_performance():
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

    model_data = joblib.load("models/heart_disease_model.pkl")
    model = model_data["model"]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nTesting Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Heart Disease Prediction - Confusion Matrix")

    plt.savefig(
        "screenshots/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    model_names = []
    model_scores = []

    print("\n5-Fold Cross-Validation Comparison:")

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        mean_score = scores.mean() * 100

        model_names.append(name)
        model_scores.append(mean_score)

        print(name, ":", round(mean_score, 2), "%")

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=model_names,
        y=model_scores
    )

    plt.xlabel("Machine Learning Models")
    plt.ylabel("Mean Cross-Validation Accuracy (%)")
    plt.title("Model Accuracy Comparison")

    plt.ylim(0, 100)
    plt.xticks(rotation=10)

    plt.savefig(
        "screenshots/model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    visualize_model_performance()
