import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict purchase
def predict_purchase(gender, age, salary):
    # Convert gender to binary (0 for Female, 1 for Male)
    gender = 1 if gender == 'Male' else 0
    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[gender, age, salary]], columns=['Gender', 'Age', 'Salary'])
    # Predict using the model
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title('Customer Purchase Prediction')

# Input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
salary = st.number_input('Salary', min_value=0, value=50000)

# Predict button
if st.button('Predict'):
    prediction = predict_purchase(gender, age, salary)
    if prediction == 1:
        st.success('The customer is likely to purchase.')
    else:
        st.success('The customer is not likely to purchase.')
