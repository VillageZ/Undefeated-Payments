# Import the required libraries
import pandas as pd
from scipy.stats import pointbiserialr

# Load the dataset
data = pd.read_csv(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default-of-credit-card-clients.csv', skiprows=1)

# Rename columns for better readability
data.columns = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 
                'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default payment next month']

# Convert MARRIAGE into a binary variable (e.g., married (MARRIAGE=1) vs. others)
data['SEX_BINARY'] = data['SEX'].apply(lambda x: 1 if x == 1 else 0)

# Calculate the point-biserial correlation between the binary MARRIAGE and default payment next month
corr, p_value = pointbiserialr(data['SEX_BINARY'], data['default payment next month'])

# Print the correlation coefficient and p-value
print(f"Point-Biserial Correlation (SEX vs. non payment): {corr}")
print(f"P-value: {p_value}")
