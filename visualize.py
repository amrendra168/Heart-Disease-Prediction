
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

    print("\nLogistic Regression Results:")
    print("Training Accuracy:", accuracy_score(y_train, model.predict(X_train)))
    print("Testing Accuracy:", accuracy_score(y_test, y_pred))

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
    plt.title("Confusion Matrix")

    plt.savefig("screenshots/confusion_matrix.png", dpi=300, bbox_inches="tight")
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

    train_scores = []
    test_scores = []

    print("\n5-Fold Cross-Validation Comparison:")

    for name, current_model in models.items():

        scores = cross_val_score(
            current_model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        model_names.append(name)
        model_scores.append(scores.mean() * 100)

        current_model.fit(X_train, y_train)

        train_accuracy = accuracy_score(
            y_train,
            current_model.predict(X_train)
        )

        test_accuracy = accuracy_score(
            y_test,
            current_model.predict(X_test)
        )

        train_scores.append(train_accuracy * 100)
        test_scores.append(test_accuracy * 100)

        print(name, "CV Accuracy:", round(scores.mean() * 100, 2), "%")
        print(name, "Training Accuracy:", round(train_accuracy * 100, 2), "%")
        print(name, "Testing Accuracy:", round(test_accuracy * 100, 2), "%")

    plt.figure(figsize=(9, 5))

    sns.barplot(x=model_names, y=model_scores)

    plt.xlabel("Machine Learning Models")
    plt.ylabel("Mean Cross-Validation Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 100)

    plt.savefig("screenshots/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    comparison_data = pd.DataFrame({
        "Model": model_names,
        "Training Accuracy": train_scores,
        "Testing Accuracy": test_scores
    })

    comparison_data = comparison_data.melt(
        id_vars="Model",
        var_name="Dataset",
        value_name="Accuracy"
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=comparison_data,
        x="Model",
        y="Accuracy",
        hue="Dataset"
    )

    plt.xlabel("Machine Learning Models")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Testing Accuracy Comparison")
    plt.ylim(0, 100)

    plt.savefig("screenshots/train_test_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    visualize_model_performance()
