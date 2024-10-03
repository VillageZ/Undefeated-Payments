import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load the dataset
data = pd.read_csv(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default-of-credit-card-clients.csv', skiprows=1)

# Define features and target variable
X = data[['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'BILL_AMT2']]
y = data['default payment next month']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and fit the Ridge regression model
ridge_model = Ridge(alpha=1.0)  # You can tune the alpha parameter for better performance
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred_ridge = ridge_model.predict(X_test)

# Convert predictions to binary based on a threshold
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_ridge]

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_ridge)
r2 = r2_score(y_test, y_pred_ridge)
accuracy = accuracy_score(y_test, y_pred_binary)

# Print results
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Accuracy: {accuracy:.4f}")
