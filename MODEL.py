import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Create sample data (replace this with your own dataset loading)

data = pd.DataFrame("Sample_Data (1).csv")  # Load your dataset here
X = data.drop('target_column', axis=1)  # Adjust the target_column
y = data['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
st.title("Model Evaluation App")

# Dropdown options for models
available_models = [
    'Linear Regression',
    'Decision Tree',
    'Random Forest',
    'Bagging'
]
selected_model = st.selectbox('Select a Model', available_models)

# Train and evaluate models
def train_and_evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

if selected_model == 'Linear Regression':
    model = LinearRegression()
elif selected_model == 'Decision Tree':
    model = DecisionTreeRegressor(random_state=42)
elif selected_model == 'Random Forest':
    model = RandomForestRegressor(random_state=42)
elif selected_model == 'Bagging':
    model = BaggingRegressor(random_state=42)

mse, mae, r2 = train_and_evaluate_model(model)

# Display evaluation results
st.subheader(f"Evaluation Results for {selected_model}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R2 Score: {r2:.2f}")
