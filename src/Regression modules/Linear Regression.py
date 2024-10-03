# Load dataset
import pandas as pd

#use skickitlearn linear regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#use skickitlearn train test split and standard scalarized data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default-of-credit-card-clients.csv', skiprows=1)
print(data)

X = data.drop(['default payment next month', 'ID', data.columns[0]], axis=1)
y = data['default payment next month']

#set variables x_train and y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Evaluate
y_pred_log = log_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {accuracy_log}")