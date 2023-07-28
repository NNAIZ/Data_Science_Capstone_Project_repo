import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import requests
import os

# Function to download the model file from Google Drive
def download_model(file_url):
    req = requests.get(file_url)
    with open('best_model.pkl', 'wb') as file:
        file.write(req.content)

# Download the model file from the provided link
file_url = 'https://drive.google.com/open?id=10KpJDZvQECn5DZhd_NiHuGuLHpdCpL3n&usp=drive_copy'
download_model(file_url)

# Get the current working directory
current_dir = os.getcwd()

# Construct the full file path to the model file
model_file_path = os.path.join(current_dir, 'best_model.pkl')

# Load the saved model
loaded_model = joblib.load(model_file_path)

# Function to preprocess the data
def preprocess_data(data):
    # Create an instance of LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to the categorical columns
    categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

# Main function to run the Streamlit app
def main():
    st.title('Car Price Prediction')

    # Create a form to input new data
    st.subheader('Enter Car Details')
    name = st.text_input('Car Model:')
    year = st.number_input('Year:', 2000, 2023, step=1)
    km_driven = st.number_input('Kilometers Driven:')
    fuel = st.selectbox('Fuel Type:', ['Petrol', 'Diesel'])
    seller_type = st.selectbox('Seller Type:', ['Individual', 'Dealer'])
    transmission = st.selectbox('Transmission:', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner:', ['First Owner', 'Second Owner', 'Third Owner'])

    # Create a DataFrame with the input data
    new_data = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })

    # Preprocess the new data
    new_data = preprocess_data(new_data)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(new_data)
    st.subheader('Predicted Selling Price:')
    st.write(f"${prediction[0]:,.2f}")

# Run the app
if __name__ == '__main__':
    main()
