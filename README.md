
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 30.09.2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Crypto Data Since 2015 (1).csv', index_col='Date', parse_dates=True)

# Perform Augmented Dickey-Fuller test for stationarity on the 'money' column
result = adfuller(data['Bitcoin (USD)'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 13
model = AutoReg(train_data['Bitcoin (USD)'], lags=lag_order)
model_fit = model.fit()

plt.figure(figsize=(10, 6))
plot_acf(data['Bitcoin (USD)'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['Bitcoin (USD)'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

mse = mean_squared_error(test_data['Bitcoin (USD)'], predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data['Bitcoin (USD)'], label='Test Data - Price of Bitcoin (USD)')
plt.plot(predictions, label='Predictions - Price of Bitcoin (USD)',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()

```
### OUTPUT:

PACF - ACF

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/e781a894-6509-4f28-9e69-d70d61c0a66d" />
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/4e2ffda3-f804-4528-8e40-d9ec48ebec11" />

PREDICTION

<img width="424" height="31" alt="image" src="https://github.com/user-attachments/assets/d3d69391-dd6b-48ca-81cc-36013a308187" />


FINIAL PREDICTION

<img width="1032" height="547" alt="image" src="https://github.com/user-attachments/assets/56a445b4-1e9a-4ce6-9d8a-1adaebcf934c" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
