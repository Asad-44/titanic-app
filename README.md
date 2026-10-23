# Titanic Passenger Survival Prediction

A machine learning web app that predicts whether a Titanic passenger would have survived, based on four passenger attributes. Built for the Machine Learning Fundamentals course (AIC354) at COMSATS University Islamabad, Lahore Campus.

**Live demo:** [https://asad-44-titanic-app-app-oaq5vi.streamlit.app/]

## Overview

The model is a Support Vector Classifier (SVC) trained on a small sample dataset. Categorical inputs are converted to numbers with label encoding, and the trained model is served through a Streamlit interface.

## Input Features

| Feature | Values |
|---|---|
| Passenger Class | First, Second, Third |
| Gender | Male, Female |
| Siblings/Spouses aboard | Zero, One, Two, Three |
| Port of Embarkation | Cherbourg, Southampton, Queenstown |

**Output:** Survived / Did not survive

## Project Structure

```
titanic-app/
├── app.py                  # Streamlit application
├── svc_trained_model.pkl   # Trained SVC model
├── requirements.txt        # Python dependencies
└── README.md
```

## Run Locally

```bash
git clone https://github.com/Asad-44/titanic-app.git
cd titanic-app
pip install -r requirements.txt
streamlit run app.py
```

## Machine Learning Workflow

1. Data collection and understanding
2. Preprocessing
3. Label encoding of inputs and output
4. Training (SVC) and saving the model
5. Testing and evaluation
6. Application phase (this web app)

## Limitations

- Trained on a small sample dataset, so predictions are illustrative rather than reliable.
- Built for learning purposes only.

## Author

Muhammad Asad Shafiq
BS Software Engineering, COMSATS University Islamabad, Lahore Campus
GitHub: [Asad-44](https://github.com/Asad-44)
