# Load required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)
#                    dtype='float64',
 #                   na_values=['', ' ', 'NA', 'NaN'])
#
data.drop('ID',axis=1)
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
#print(data)

# Select the columns you are interested in for pairplot
columns_to_plot = data[['default payment next month', 'SEX','EDUCATION','LIMIT_BAL','PAY_0','PAY_2','BILL_AMT1','BILL_AMT2']]

#print(columns_to_plot)

# Generate the pairplot
sns.pairplot(columns_to_plot, hue='default payment next month')

#Display the plot
plt.show()
