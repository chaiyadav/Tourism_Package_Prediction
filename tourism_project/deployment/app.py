import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="chaitanya-yadav/Tourism-Package-Prediction", filename="best_tourism_package_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool to help staff identifying the potential customers efficiently.")
st.write("Kindly enter the potential Customer details to check whether they are likely to buy any travel package.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Contact Type (Method by which the customer was contacted)", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City category (development, population, and living standards)", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("occupation", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip", min_value=0, max_value=6, value=1)
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer", min_value=3, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Average number of trips the customer takes annually", min_value=0, max_value=30, value=3)
Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("Owns a car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer", min_value=0, max_value=3, value=1)
Designation = st.selectbox("Designation in their current organization", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Gross monthly income of the customer", min_value=0, max_value=100000, value=5000)

st.caption("Customer Interaction Data")

PitchSatisfactionScore = st.selectbox("Sales pitch satisfaction score.", options=list(range(1, 5)))
ProductPitched = st.selectbox("Type of product pitched", ["Basic", "Standard", "Deluxe", "King", "Super Deluxe"])
NumberOfFollowups = st.number_input("Post-pitch follow-up count", min_value=0, max_value=10, value=1)
DurationOfPitch = st.number_input("Sales pitch duration", min_value=5,  value=10)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': 1 if CityTier == "Tier 1" else 2 if CityTier == "Tier 2" else 3,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation' : Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "not likely to purchase tourism package" if prediction == 1 else "likely to Purchase Tourism Package"
    st.write(f"Based on the information provided, the customer is  {result}.")
