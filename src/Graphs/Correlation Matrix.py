# Load required libraries
import pandas as pd

# Load the dataset
data = pd.read_csv(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default-of-credit-card-clients.csv', skiprows=1)

# Rename columns for better readability
data.columns = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 
                'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default payment next month']

# Drop non-numeric columns, like 'ID', since they don't contribute to correlation analysis
numeric_data = data.drop(columns=['ID','SEX'])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Define the file path where the CSV will be saved
output_file_path = r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\src\Graphs\correlation_matrix.xlsx'

# Save the correlation matrix to a CSV file
correlation_matrix.to_excel(output_file_path, index=True)

# Optionally, print a confirmation message
print(f"Correlation matrix saved to {output_file_path}")