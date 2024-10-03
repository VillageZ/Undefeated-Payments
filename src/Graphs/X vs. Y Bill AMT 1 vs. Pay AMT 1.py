import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
import pandas as pd

data = pd.read_csv(r'C:\Users\zclar\OneDrive\Documents\Python-Projects\Credit-Project\default-of-credit-card-clients.csv', skiprows=1)

scaler = StandardScaler()
X_PLOT = scaler.fit_transform(data['BILL_AMT1'])
Y_PLOT = scaler.transform(data['PAY_AMT1'])

Right_max = data['X_PLOT'].max()
Top_max = data['Y_PLOT'].max()

import matplotlib.pyplot as plt

plt.scatter(data['X_PLOT'],data['Y_PLOT'])
plt.xlim(left = 0, right=Right_max *1.1)
plt.ylim(bottom = 0, top=Top_max *1.1)
plt.xlabel('BILL_AMT1')
plt.ylabel('PAY_AMT1')
plt.title('BILL_AMT1 vs PAY_AMT1')
plt.show()