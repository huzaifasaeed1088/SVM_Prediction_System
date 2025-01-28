import pandas as pd
import pickle
import os

import streamlit as st
import pandas as pd
import pickle
from f2020266522 import preprocess_data  # Assuming this contains data preprocessing logic

# Load the pre-trained model
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app
st.title("Customer Purchase Prediction")
st.write("Predict whether a customer will purchase based on gender, age, and salary.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25, step=1)
salary = st.number_input("Salary", min_value=0, value=50000, step=1000)

# Preprocess inputs
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Salary": [salary]
    })

    # Preprocess input_data if needed
    processed_data = preprocess_data(input_data)  # Implement this in your main file

    # Predict
    prediction = model.predict(processed_data)
    result = "will purchase" if prediction[0] == 1 else "will not purchase"
    st.write(f"The customer {result}.")

# Display CSV for demonstration
st.subheader("Sample Data")
user_data = pd.read_csv("user_data.csv")
st.dataframe(user_data)
