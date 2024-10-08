# Load required libraries
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)

# Drop non-numeric columns, like 'ID' and 'SEX', since they don't contribute to VIF calculation
numeric_data = data.drop(columns=['ID', 'SEX', 'default payment next month'])

# Convert all columns to float
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# Function to calculate VIF
def calculate_vif(df, features):
    vif = pd.DataFrame()
    vif["Variable"] = features
    vif["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif

# Calculate VIF for the numeric features
vif_data = calculate_vif(numeric_data, numeric_data.columns)

# Specify the path to save the Excel file
output_path = r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\src\Graphs\vif_values.xlsx'

# Write VIF data to an Excel file
vif_data.to_excel(output_path, index=False)

# Print a message indicating the file has been saved
print(f"VIF values have been saved to {output_path}")
