import streamlit as st
import pickle
import numpy as np

# Function to load the car price prediction model and make predictions
def predict_car_price(features):
    # Load the trained model from the pickle file
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make predictions using the loaded model
    price_prediction = model.predict(features)
    return price_prediction[0]

# Streamlit app
def main():
    # Title of the web application
    st.title('Car Price Prediction Model')

    # Create a form for user input
    st.subheader('Enter Car Details')
    year = st.number_input('Year of Manufacture', 2000, 2023, 2007)
    km_driven = st.number_input('Kilometers Driven', 0, 1000000, 70000)
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    # Create a numpy array with the user input
    user_data = np.array([[year, km_driven]])

    # Make predictions
    predicted_price = predict_car_price(user_data)

    # Display the predictions
    st.subheader('Car Price Prediction')
    st.write('User Input Data:')
    st.write(f'Year of Manufacture: {year}')
    st.write(f'Kilometers Driven: {km_driven}')
    st.write(f'Seller Type: {seller_type}')
    st.write(f'Transmission: {transmission}')
    st.write(f'Owner: {owner}')
    st.write(f'Predicted Car Price: ${predicted_price:,.2f}')

    # Display success message
    st.success('Car price prediction completed.')

if __name__ == '__main__':
    main()
