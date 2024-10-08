# Load dataset
import pandas as pd

#use skickitlearn linear regression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#use skickitlearn train test split and standard scalarized data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)
print(data)

X = data[['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'BILL_AMT2']]

y = data['default payment next month']

#set variables x_train and y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Evaluate the model
print(f"Accuracy: {accuracy_rf:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

print(feature_importance_df.sort_values(by='Importance', ascending=False))
