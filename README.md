# Ex.No: 6               HOLT WINTERS METHOD
### Date: 18.05.2026

### AIM:
To analyze monthly sales trends using time series decomposition and forecast future sales using the Holt-Winters model evaluated by RMSE.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('/content/stock_data (1).csv')
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
data_monthly = data['Stock_1'].resample('MS').mean()

scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1,1)).flatten(),
    index=data_monthly.index
)

scaled_data = scaled_data + 1

scaled_data.plot(figsize=(10,5), title='Scaled Data',color='purple')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.show()

decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=4
)

decomposition.plot()
plt.show()

train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal=None
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(12,6),
                     label='Train Data',
                     color='green')

test_data.plot(ax=ax,
               label='Test Data',
               color='blue')

test_predictions_add.plot(ax=ax,
                          label='Predictions',
                          color='red')

ax.legend()
ax.set_title('Holt-Winters Prediction')
plt.show()

mae = mean_absolute_error(test_data, test_predictions_add)

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))

print("MAE :", mae)
print("RMSE :", rmse)

final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal=None
).fit()

final_predictions = final_model.forecast(steps=12)

ax = scaled_data.plot(figsize=(12,6),
                      label='Historical Data')

final_predictions.plot(ax=ax,
                       label='Future Forecast',
                       color='orange')

ax.legend()

ax.set_title('Future Forecast using Holt-Winters')

ax.set_xlabel('Date')
ax.set_ylabel('Scaled Stock Price')

plt.show()

print(final_predictions)
```

### OUTPUT:

<img width="882" height="497" alt="image" src="https://github.com/user-attachments/assets/b0aa2f0d-43e4-4884-8692-bbbcede36da4" />

<img width="665" height="477" alt="image" src="https://github.com/user-attachments/assets/ae2adbf4-379d-41b2-984a-6f6c6b008b9c" />

<img width="1041" height="570" alt="image" src="https://github.com/user-attachments/assets/5f2ff077-db3a-4118-bd5d-bff9fd7303c3" />

<img width="1056" height="572" alt="image" src="https://github.com/user-attachments/assets/82be448e-c621-4ddf-a859-41d2c7e7acd8" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
