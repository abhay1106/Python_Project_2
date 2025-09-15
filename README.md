# Walmart Sales Forecasting Project

This project focuses on forecasting weekly sales for Walmart stores using **SARIMA** (Seasonal ARIMA) models. The goal is to analyze historical sales data, detect trends and seasonality, and predict future sales.

---

## 1. Project Overview

- **Dataset:** Walmart weekly sales data (`Walmart_dataset.csv`)
- **Objective:** Forecast weekly sales for each store
- **Techniques Used:** 
  - Time Series Analysis
  - SARIMA (Seasonal ARIMA)
  - Data Visualization
  - Model Evaluation (RMSE)

---

## 2. Dataset Description

The dataset should contain at least the following columns:

| Column | Description |
|--------|-------------|
| `store` | Store ID |
| `date`  | Week-ending date |
| `sales` | Weekly sales for the store |

---
## 3. Required Python Packages:

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

## 4. Exploratory Data Analysis
```python
# Check for missing values
walmart_df.isnull().sum()

# Summary statistics
walmart_df.describe()

# Sales trends over time
sns.lineplot(data=walmart_df, x='Date', y='Weekly_Sales', hue='Store')
```
## 5. Correlation Analysis
```python
# Correlation with unemployment, CPI, temperature
walmart_df.corr()['Weekly_Sales']
```
## 6. Features

- Forecast sales for individual stores
- Identify trends, seasonality, and patterns
- Evaluate model performance using RMSE
- Visualize historical vs forecasted sales

---

## 7. How to Run the Project
### **1. Prepare the Dataset**

Place the dataset Walmart_dataset.csv in the project directory.

Ensure the dataset contains at least the following columns:

store – Store ID

date – Week-ending date

sales – Weekly sales for each store

### **2. Load the Dataset**
```python
import pandas as pd
# Load Walmart sales data
Walmart_dataset = pd.read_csv("Walmart_dataset.csv")

# Convert date column to datetime type
Walmart_dataset['date'] = pd.to_datetime(Walmart_dataset['date'])
```
### **3. Filter Data by Store**
Forecasting is done individually for each store. For example, to analyze Store 1:
```python
store1_data = Walmart_dataset[Walmart_dataset['store'] == 1]
store1_data.set_index('date', inplace=True)
store1_data = store1_data['sales']
```

### **4. Visualize Historical Sales**
Plot the sales to understand trends and seasonality:
```python
import matplotlib.pyplot as plt

store1_data.plot(figsize=(12,6), title="Store 1 Weekly Sales")
plt.show()
```
### **5. Check Stationarity**
Perform the Augmented Dickey-Fuller (ADF) test:
```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(store1_data)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
```
### **6. Train-Test Split**
Split the data into training and testing sets:
```python
train = store1_data[:-36]  # Use all data except last 36 weeks
test = store1_data[-36:]   # Last 36 weeks for testing
```
### **7. Hyperparameter Tuning for SARIMA**
Try different combinations of (p, d, q) and (P, D, Q, s) to find the best model:
```python
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in pdq]  # s=52 for weekly data

best_rmse = float("inf")
best_params = None

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = SARIMAX(train, order=param, seasonal_order=seasonal_param,
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            pred = results.predict(start=test.index[0], end=test.index[-1])
            rmse = np.sqrt(mean_squared_error(test, pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (param, seasonal_param)
        except:
            continue

print(f"Best Parameters: {best_params}, RMSE: {best_rmse}")
```
### **8. Train the Final Model**
Use the best parameters to train the SARIMA model:
```python
best_order, best_seasonal_order = best_params
final_model = SARIMAX(store1_data, order=best_order, seasonal_order=best_seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False)
final_results = final_model.fit()
```
### **9. Forecast Future Sales**
Forecast for the next 36 weeks:
```python
forecast = final_results.get_forecast(steps=36)
forecast_values = forecast.predicted_mean

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(store1_data, label="Historical Sales")
plt.plot(forecast_values, label="Forecasted Sales", color='red')
plt.title("Store 1 Sales Forecast")
plt.legend()
plt.show()
```
## 8. Repeated for Other Stores
- Repeat steps 3-9 for each store (Store 2, Store 45, etc.).

- We can also automate this process using a loop over unique store IDs.

## 9. Results

- The model successfully forecasts weekly sales for multiple Walmart stores.

- RMSE values indicate the accuracy of forecasts for each store.

- Seasonal trends (weekly/monthly patterns) were captured effectively using SARIMA.

- Visualizations highlight how forecasts follow historical sales patterns.

## 10. Key Learnings

- Time series forecasting can reveal trends and seasonal patterns in retail sales.

- SARIMA models are effective for handling both trend and seasonality in weekly data.

- Model performance is highly dependent on proper selection of SARIMA parameters.

- Visualizing forecasts alongside actual sales helps to validate and communicate results.

- Data-driven forecasting can support inventory management and strategic planning in retail.

## 11. Conclusion

- The project demonstrates a complete workflow for retail sales forecasting using Python.

- SARIMA models are capable of producing accurate forecasts and understanding seasonal patterns.

- Forecasting enables better decision-making for inventory and sales strategy.

## 12. Future improvements could include:

- Incorporating external factors (holidays, promotions)

- Using machine learning models like XGBoost or LSTM for higher accuracy

- Automating model selection and parameter tuning for multiple stores
