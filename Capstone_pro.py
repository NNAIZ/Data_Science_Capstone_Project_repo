import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# Define a function to preprocess new data
def preprocess_new_data(new_data):
    label_encoder = LabelEncoder()

    # Apply label encoding to the categorical columns in new_data
    categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']

    for column in categorical_columns:
        new_data[column] = label_encoder.fit_transform(new_data[column])

    return new_data

# Define the Streamlit app
def main():
    st.title('Car Price Prediction')

    model_file = 'best_model.pkl'

    # Check if the model file exists in the same directory
    if not os.path.exists(model_file):
        st.error(f"Model file '{model_file}' not found in the same directory. "
                 f"Please make sure the model file is in the same directory as this script.")
        return

    try:
        # Load the saved model
        with open(model_file, 'rb') as file:
            loaded_model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

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
    new_data_encoded = preprocess_new_data(new_data)

    # Use the loaded model to make predictions on the new data
    predictions = loaded_model.predict(new_data_encoded)

    # Display the predicted selling price
    st.subheader('Predicted Selling Price')
    st.write(f'₹ {predictions[0]:,.2f}')

if __name__ == '__main__':
    main()
