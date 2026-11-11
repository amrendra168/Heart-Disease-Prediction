# Heart Disease Prediction

A beginner-friendly Machine Learning project that uses Logistic Regression to classify heart disease records using a dataset of medical attributes.

## Project Overview

This project demonstrates how Machine Learning can be used to predict whether a record belongs to the heart disease class or the no heart disease class.

The model is trained using a Kaggle heart disease dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Machine Learning Algorithm

**Logistic Regression**

The model uses:
- StandardScaler for feature scaling
- Logistic Regression for classification
- Train-test split for model evaluation

## Dataset Information

- Total original records: 1,025
- Records after removing duplicates: 302
- Input features: 13
- Target column: `target`

Target values:
- `0` – No Heart Disease Class
- `1` – Heart Disease Class

## Model Performance

| Metric | Result |
|---|---:|
| Training Accuracy | 85.89% |
| Testing Accuracy | 80.33% |
| Precision (Class 1) | 80.00% |
| Recall (Class 1) | 84.85% |
| F1-score (Class 1) | 82.00% |

## Project Structure

```text
Heart-Disease-Prediction/
│
├── dataset/
│   └── heart.csv
│
├── models/
│   └── heart_disease_model.pkl
│
├── main.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

2. Open the project folder:

```bash
cd Heart-Disease-Prediction
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment on Windows:

```powershell
venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

Choose an option from the menu:

1. Train Model
2. Make Prediction
0. Exit

## Future Improvements

- Add a simple web interface using Streamlit.
- Compare additional machine learning algorithms.
- Improve model evaluation using cross-validation.
- Add visualizations to explain predictions.

## Disclaimer

This project is developed for educational purposes only. Its predictions are based on a dataset and must not be used for medical diagnosis or treatment decisions.