from scipy.stats import shapiro
import pandas as pd
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)

for col in data.columns:
    stat, p = shapiro(data[col])
    print(f'Column: {col}, p-value: {p}')
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Not Probably Gaussian')