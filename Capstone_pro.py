# Required libraries
import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Function to preprocess the car dataset
def preprocess_car_data(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)

    # Convert categorical features to numerical form using one-hot encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Separate the features (X) and target (y)
    X = data_encoded.drop(columns=['selling_price'])
    y = data_encoded['selling_price']

    return X, y

# Function to train the car price prediction model and save it
def train_and_save_model(X_train, y_train):
    # Train the model (replace this with your actual model training process)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model using pickle
    with open('car_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Function to load the car price prediction model and make predictions
def predict_car_price(features):
    # Load the trained model from the pickle file
    with open('car_price_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make predictions using the loaded model
    price_prediction = model.predict(features)
    return price_prediction

# Sample data for demonstration purposes
# Replace this with your actual dataset for model training
car_data = {
    'name': ['Maruti 800 AC', 'Hyundai Verna 1.6 SX'],
    'year': [2007, 2012],
    'km_driven': [70000, 100000],
    'fuel': ['Petrol', 'Diesel'],
    'seller_type': ['Individual', 'Individual'],
    'transmission': ['Manual', 'Manual'],
    'owner': ['First Owner', 'First Owner'],
    'selling_price': [150000, 500000]
}

# Create a DataFrame from the sample data
car_df = pd.DataFrame(car_data)

# Preprocess the car dataset
X_train, y_train = preprocess_car_data(car_df)

# Train the car price prediction model and save it
train_and_save_model(X_train, y_train)

# Streamlit app
def main():
    # Title of the web application
    st.title('Car Price Prediction Model')

    # Display the sample data
    st.subheader('Sample Data for Model Training')
    st.write('Features (X_train):')
    st.write(X_train)
    st.write('Target (y_train - Car Prices in USD):')
    st.write(y_train)

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

    # Preprocess the user data
    X_user, _ = preprocess_car_data(user_data)

    # Make predictions
    predicted_price = predict_car_price(X_user)

    # Display the predictions
    st.subheader('Car Price Prediction')
    st.write('Features (X_new):')
    st.write(X_user)
    st.write(f'Predicted Car Price: ${predicted_price[0]:,.2f}')

    # Display success message
    st.success('Car price prediction completed.')

if __name__ == '__main__':
    main()

