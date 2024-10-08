# Load dataset
import pandas as pd
import numpy as np

# Use scikit-learn Logistic Regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Use scikit-learn train_test_split and StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and prepare data
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)
print(data.dtypes)

X = data[['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'BILL_AMT2']]
y = data['default payment next month']

# Define data types
ordinal_cols = ['PAY_0']
continuous_cols = ['BILL_AMT1', 'BILL_AMT2', 'AGE', 'LIMIT_BAL']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(), ordinal_cols),
        ('continuous', StandardScaler(), continuous_cols),
            ]
)

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(), ordinal_cols),
        ('continuous', StandardScaler(), continuous_cols),
                   ]
)

# Create a pipeline that combines preprocessing with the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=69))  # Increase max_iter if needed
])

# Set aside the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict
y_proba = pipeline.predict_proba(X_test)[:, 1] 
y_pred = pipeline.predict(X_test)

# Calculate errors
errors = abs(y_pred - y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Evaluate the model
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))