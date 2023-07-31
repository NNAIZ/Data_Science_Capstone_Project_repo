import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

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
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model using joblib
    joblib.dump(model, 'best_model.pkl')

# Function to load the car price prediction model and make predictions
def predict_car_price(features):
    # Load the trained model from the joblib file
    loaded_model = joblib.load('best_model.pkl')

    # Make predictions using the loaded model
    price_prediction = loaded_model.predict(features)
    return price_prediction

# Streamlit app
def main():
    # Title of the web application
    st.title('Car Price Prediction Model')

    # Upload the CSV file
    uploaded_file = st.file_uploader("/content/CAR DETAILS (1v", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        car_df = pd.read_csv(uploaded_file)

        # Preprocess the car dataset
        X_train, y_train = preprocess_car_data(car_df)

        # Train the car price prediction model and save it
        train_and_save_model(X_train, y_train)

        # Display the sample data used for model training
        st.subheader('Sample Data for Model Training')
        st.write('Features (X_train):')
        st.write(X_train)
        st.write('Target (y_train - Car Prices in USD):')
        st.write(y_train)

        # Select 20 random data points from the original dataset
        random_subset = car_df.sample(n=20, random_state=42)

        # Remove the 'selling_price' column from the random subset
        subset_for_prediction = random_subset.drop('selling_price', axis=1)

        # Preprocess the subset for prediction (if needed)
        subset_for_prediction_encoded = pd.get_dummies(subset_for_prediction, drop_first=True)

        # Make predictions
        predicted_prices = predict_car_price(subset_for_prediction_encoded)

        # Add the predictions as a new column in the random subset
        random_subset['predicted_selling_price'] = predicted_prices

        # Display the random subset with predictions
        st.subheader('Random Subset with Predictions')
        st.write(random_subset)

if __name__ == '__main__':
    main()

