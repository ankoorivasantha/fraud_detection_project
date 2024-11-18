import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Custom CSS to set the colors
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;  /* Light Blue Background */
    }
    .stButton>button {
        background-color: #4CAF50;  /* Green Button */
        color: white;
    }
    .stSuccess {
        color: green;
    }
    .stError {
        color: red;
    }
    .stTitle {
        color: #004d99;  /* Blue title text */
    }
    .stHeader {
        color: #004d99;  /* Blue header text */
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Fastag Fraud Detection")

# User input
st.header("Enter Key Transaction Details:")
transaction_amount = st.number_input("Transaction Amount (₹)", min_value=0)
amount_paid = st.number_input("Amount Paid (₹)", min_value=0)
vehicle_type = st.selectbox("Vehicle Type", ['Bus', 'Car', 'Motorcycle', 'Truck'])
lane_type = st.selectbox("Lane Type", ['Express', 'Regular'])
vehicle_speed = st.number_input("Vehicle Speed (km/h)", min_value=0)

# Map inputs to numerical values
vehicle_type_map = {'Bus': 0, 'Car': 1, 'Motorcycle': 2, 'Truck': 3}
lane_type_map = {'Express': 0, 'Regular': 1}

# Convert user inputs into a DataFrame
features = pd.DataFrame({
    'Transaction_Amount': [transaction_amount],
    'Amount_paid': [amount_paid],
    'Vehicle_Type': [vehicle_type_map[vehicle_type]],
    'Lane_Type': [lane_type_map[lane_type]],
    'Vehicle_Speed': [vehicle_speed],
})

# Predict
if st.button("Check for Fraud"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("Fraud Detected!")
    else:
        st.success("No Fraud Detected.")
