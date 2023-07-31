# Required libraries
import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to train the model and save it
def train_and_save_model(X_train, y_train):
    # Train the model (replace this with your actual model training process)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model using pickle
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Sample data for demonstration purposes
# Replace this with your actual dataset for model training
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Train the model and save it
train_and_save_model(X_train, y_train)

# Streamlit app
def main():
    # Title of the web application
    st.title('Simple Linear Regression Model Trainer')

    # Display the sample data
    st.subheader('Sample Data for Model Training')
    st.write('X_train:')
    st.write(X_train)
    st.write('y_train:')
    st.write(y_train)

    # Train and save the model using the sample data
    train_and_save_model(X_train, y_train)

    # Display success message
    st.success('Model training completed and the model has been saved as "best_model.pkl".')

if __name__ == '__main__':
    main()

