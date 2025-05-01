import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 House Price Prediction")
st.write("Enter the details below to predict house price")

def load_and_train_model():
    # Load data
    data = pd.read_csv('House Price Prediction ML/housing.csv')
    
    # Prepare features
    X = data.drop(['median_house_value', 'ocean_proximity'], axis=1)
    y = data['median_house_value']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

# Load or train model
@st.cache_resource
def get_model():
    return load_and_train_model()

model, features = get_model()

# Create input form
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
    latitude = st.number_input("Latitude", value=37.88, format="%.4f")
    housing_median_age = st.number_input("Housing Median Age", value=41.0, format="%.1f")
    total_rooms = st.number_input("Total Rooms", value=880.0, format="%.1f")

with col2:
    total_bedrooms = st.number_input("Total Bedrooms", value=129.0, format="%.1f")
    population = st.number_input("Population", value=322.0, format="%.1f")
    households = st.number_input("Households", value=126.0, format="%.1f")
    median_income = st.number_input("Median Income", value=8.3252, format="%.4f")

# Predict button
if st.button("Predict Price"):
    # Create input array
    input_data = pd.DataFrame([[
        longitude, latitude, housing_median_age, total_rooms,
        total_bedrooms, population, households, median_income
    ]], columns=features)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f"Predicted House Price: ${prediction:,.2f}")

# Add information about the project
st.markdown("""
### About this Project
This house price prediction model uses machine learning to estimate house prices based on various features including:
- Geographic location (longitude and latitude)
- Housing median age
- Number of rooms and bedrooms
- Local population and households
- Median income of the area

The model is trained on California housing data and provides estimates in USD.
""")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 