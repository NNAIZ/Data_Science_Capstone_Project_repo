import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data for demonstration purposes
# Replace this with your actual dataset for model training
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Train the model (replace this with your actual model training process)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('best_model.pkl', 'wb') as file:
    pickle.dump(model, file)
