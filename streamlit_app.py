import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 House Price Prediction")

MedInc = st.number_input("Median Income", min_value=0.5, max_value=15.0, value=3.0)
HouseAge = st.number_input("House Age", min_value=1.0, max_value=52.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=2.0, max_value=10.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=1.0, max_value=5.0, value=2.0)
Population = st.number_input("Population", min_value=100.0, max_value=5000.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=1.0, max_value=5.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0)
Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-122.0)

if st.button("Predict"):
    data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                      Population, AveOccup, Latitude, Longitude]])

    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    price_dollars = result[0] * 100000
    formatted_price = f"{price_dollars:,.0f}"

    st.success(f"Estimated House Price: ${formatted_price}")







