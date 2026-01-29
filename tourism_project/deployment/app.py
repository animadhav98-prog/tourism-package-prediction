import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="ani-maddy98/prediction-model", filename="best_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for Visit With Us staff that predicts whether a given customer will be likely to purchase our Wellness Package. This will help sales teams spend their time on the customers most likely to convert.")
st.write("Kindly fill out customer details to check whether they are likely to purchase the package.")

# Collect user input
Age = st.number_input("Customer's Age", min_value=18, max_value=80, value=40)
MonthlyIncome = st.number_input("Customer's Gross Monthly Income", min_value=1000, max_value=100000, value=25000)
NumberOfPersonVisiting = st.number_input("Total # of people accompanying the customer",min_value = 0,max_value = 10, value=2)
NumberOfTrips = st.number_input("Avg # of trips taken by customer annually",min_value = 1,max_value = 30, value=4)
NumberOfChildrenVisiting = st.number_input("No. of children below age 5 accompanying the customer",min_value = 0,max_value = 5, value=1)
CityTier = st.selectbox("City category based on development, population, and living standards", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Customer's occupation",["Free Lancer","Large Business","Salaried","Small Business"])
Gender = st.selectbox("Customer's gender",["Female","Male"])
ProductPitched = st.selectbox("Type of product pitched to the customer",["Basic","Deluxe","King","Standard","Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred hotel rating by the customer",[1,2,3,4,5])
MaritalStatus = st.selectbox("Marital status of customer",["Divorced","Married","Single"])
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Has Own Car?", ["Yes", "No"])
Designation = st.selectbox("Customer designation",["AVP","Executive","Manager","Senior Manager","VP"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'MonthlyIncome': MonthlyIncome,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting
    'CityTier': 1 if CityTier == "Tier 1" else (2 if CityTier == "Tier 2" else 3),
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.50

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "will purchase" if prediction == 1 else "will not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
