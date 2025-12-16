import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("House Price Prediction")

MedInc = st.number_input("Median Income", min_value=0.0)
HouseAge = st.number_input("House Age", min_value=0.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0)
Population = st.number_input("Population", min_value=0.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0)
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict"):
    data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                      Population, AveOccup, Latitude, Longitude]])
    data = scaler.transform(data)
    result = model.predict(data)

    st.success(f"Estimated House Price: ${result[0]:.2f}")
