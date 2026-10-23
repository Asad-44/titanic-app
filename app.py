import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Rebuild the same label encoders used in the notebook
def make_encoder(values):
    enc = LabelEncoder()
    enc.fit(values)
    return enc

pclass_enc = make_encoder(["First", "Second", "Third"])
gender_enc = make_encoder(["Male", "Female"])
sibling_enc = make_encoder(["Zero", "One", "Two", "Three"])
embarked_enc = make_encoder(["Southampton", "Cherbourg", "Queenstown"])
survived_enc = make_encoder(["Yes", "No"])

model = joblib.load("svc_trained_model.pkl")

st.title("Titanic Passenger Survival Prediction")

pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Male", "Female"])
sibling = st.selectbox("Siblings/Spouses aboard", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Southampton", "Queenstown"])

if st.button("Predict"):
    features = pd.DataFrame({
        "PClass": [pclass_enc.transform([pclass])[0]],
        "Gender": [gender_enc.transform([gender])[0]],
        "Sibling": [sibling_enc.transform([sibling])[0]],
        "Embarked": [embarked_enc.transform([embarked])[0]],
    })
    pred = model.predict(features)[0]
    result = survived_enc.inverse_transform([pred])[0]
    if result == "Yes":
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did not survive")