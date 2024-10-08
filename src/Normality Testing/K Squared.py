from scipy.stats import normaltest
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)

for col in data.columns:
    stat, p = normaltest(data[col])
    print(f'Column: {col}, p-value: {p}')
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Not probably Gaussian')

data.hist(figsize=(10, 10))
plt.show()