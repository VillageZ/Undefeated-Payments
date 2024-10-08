# Import the required libraries
import pandas as pd
from scipy.stats import pointbiserialr

# Load the dataset
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)

# Convert MARRIAGE into a binary variable (e.g., married (MARRIAGE=1) vs. others)
data['SEX_BINARY'] = data['SEX'].apply(lambda x: 1 if x == 1 else 0)

# Calculate the point-biserial correlation between the binary MARRIAGE and default payment next month
corr, p_value = pointbiserialr(data['SEX_BINARY'], data['default payment next month'])

# Print the correlation coefficient and p-value
print(f"Point-Biserial Correlation (SEX vs. non payment): {corr}")
print(f"P-value: {p_value}")
