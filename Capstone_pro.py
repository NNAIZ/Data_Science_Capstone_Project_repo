import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Function to preprocess the car dataset
def preprocess_car_data(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)

    # Convert categorical features to numerical form using label encoding
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    return data

# Load the saved model
loaded_model = joblib.load('best_model.pkl')

# Streamlit app
def main():
    # Title of the web application
    st.title('Car Price Prediction Model')

    # Create a form for user input
    st.subheader('Enter Car Details')
    name = st.text_input('Car Name', 'Maruti 800 AC')
    year = st.number_input('Year of Manufacture', 2000, 2023, 2007)
    km_driven = st.number_input('Kilometers Driven', 0, 1000000, 70000)
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })

    # Preprocess the user input data for prediction
    user_data_encoded = preprocess_car_data(user_data)

    # Use the loaded model to make predictions
    predicted_price = loaded_model.predict(user_data_encoded)

    # Display the predictions
    st.subheader('Car Price Prediction')
    st.write('User Input Data:')
    st.write(user_data)
    st.write(f'Predicted Car Price: ${predicted_price[0]:,.2f}')

    # Display success message
    st.success('Car price prediction completed.')

if __name__ == '__main__':
    main()


