import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import gdown

# Define a function to download the model file from Google Drive using gdown
def download_model():
    # ... (Your existing code for downloading the model)

# Define a function to preprocess new data
def preprocess_new_data(new_data):
    # ... (Your existing code for preprocessing new data)

# Define the Streamlit app
def main():
    st.title('Car Price Prediction')

    # Download the model file from Google Drive (Your existing code)
    model_file = download_model()

    if model_file is None:
        return

    try:
        # Load the saved model
        loaded_model = joblib.load(model_file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    # Add a file uploader to allow users to upload CSV data
    st.subheader('Upload CSV File for Prediction')
    uploaded_file = st.file_uploader("https://drive.google.com/open?id=1tVlnZVV6c5uC7MiFo6Ya0Sdb9OmPHC_N&usp=drive_copy", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file into a DataFrame
            user_data = pd.read_csv(uploaded_file)

            # Preprocess the user data
            user_data_encoded = preprocess_new_data(user_data)

            # Use the loaded model to make predictions on the user data
            predictions = loaded_model.predict(user_data_encoded)

            # Display the predicted selling prices for the user data
            st.subheader('Predicted Selling Prices')
            st.write(predictions)

        except Exception as e:
            st.error(f"Error processing the uploaded data: {e}")
            return

    # Continue with the existing code for taking user input and making predictions
    st.subheader('Enter Car Details')
    # ... (Your existing code for taking user input and making predictions)

if __name__ == '__main__':
    main()
