# Load dataset
import pandas as pd
import numpy as np

# Use scikit-learn Random Forest Regressor instead of Classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Use scikit-learn train_test_split and StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and prepare data
data = pd.read_csv(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default-of-credit-card-clients.csv', skiprows=1)
print(data)

X = data.drop(['default payment next month', 'ID', data.columns[0]], axis=1)
y = data['default payment next month'].astype(float)  # Treat y as continuous (0.0 and 1.0)

# Define data types
ordinal_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'MARRIAGE', 'EDUCATION']
binary_cols = ['SEX']
continuous_cols = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                   'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(), ordinal_cols),
        ('continuous', StandardScaler(), continuous_cols),
        ('binary', 'passthrough', binary_cols)
    ]
)

# Create a pipeline that combines preprocessing with the regression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=69))  # Use Regressor instead of Classifier
])

# Set aside the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Calculate errors
errors = abs(y_pred - y_test)

# Calculate accuracy metrics for regression
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Evaluate the model
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
