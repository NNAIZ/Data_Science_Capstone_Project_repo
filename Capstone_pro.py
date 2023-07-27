import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import requests

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

    model_url = "https://drive.google.com/open?id=10KpJDZvQECn5DZhd_NiHuGuLHpdCpL3n&usp=drive_copy"

    # Download the model file from Google Drive
    try:
        response = requests.get(model_url, verify=False)
        with open('best_model.pkl', 'wb') as f:
            f.write(response.content)
    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        return

    try:
        # Load the saved model
        loaded_model = joblib.load('best_model(3).pkl')
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
