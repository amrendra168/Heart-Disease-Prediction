
import pandas as pd
import joblib


def predict_heart_disease():
    model_data = joblib.load("models/heart_disease_model.pkl")

    model = model_data["model"]
    features = model_data["features"]

    print("\nEnter Patient Details:")

    input_data = []

    for feature in features:
        value = float(input(f"Enter {feature}: "))
        input_data.append(value)

    new_data = pd.DataFrame([input_data], columns=features)

    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)

    print("\nPrediction Result:")

    if prediction[0] == 1:
        print("Dataset Prediction: Heart Disease Class (1)")
    else:
        print("Dataset Prediction: No Heart Disease Class (0)")

    print(f"Class 0 Probability: {probability[0][0] * 100:.2f}%")
    print(f"Class 1 Probability: {probability[0][1] * 100:.2f}%")


if __name__ == "__main__":
    predict_heart_disease()
