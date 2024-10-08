import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
import pandas as pd

# Load dataset
data = pd.read_excel(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\data\default of credit card clients.xlsx',
                    skiprows=1)


# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on both columns together and then transform
X = data[['BILL_AMT1', 'PAY_AMT1']]
scaler.fit(X)
X_scaled = scaler.transform(X)

# Separate the scaled values for plotting
X_PLOT = X_scaled[:, 0]  # Scaled BILL_AMT1
Y_PLOT = X_scaled[:, 1]  # Scaled PAY_AMT1

# Get the max values for limits
Right_max = X_PLOT.max()
Top_max = Y_PLOT.max()

# Create scatter plot
plt.scatter(X_PLOT, Y_PLOT)
plt.xlim(left=0, right=Right_max * 1.1)
plt.ylim(bottom=0, top=Top_max * 1.1)
plt.xlabel('BILL_AMT1')
plt.ylabel('PAY_AMT1')
plt.title('BILL_AMT1 vs PAY_AMT1')
plt.show()
