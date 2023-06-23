# The Demand Forecasting Project

The demand forecasting project follows a structured approach with four main sections: data wrangling, exploratory data analysis (EDA), modelling, and conclusion.

## Data Wrangling:
The script begins by importing the necessary libraries and setting up the environment.
It loads the historical demand data from a CSV file into a pandas DataFrame.
It performs some data cleaning and manipulation steps, such as renaming columns and creating additional columns for year, month, and day.
The time series data is aggregated at a monthly level, summing up the quantities for each month.

## Exploratory Data Analysis (EDA):
The script visualizes the data using various plots to gain insights into the underlying patterns and trends.
It generates a line plot showing the original demand data, rolling mean, and rolling standard deviation.
It performs time series decomposition using the seasonal decomposition of time series (STL) method and visualizes the trend, seasonal, and residual components.
The script also conducts the Dickey-Fuller test to check the stationarity of the time series. 

## Modelling:
The script applies the SARIMA (Seasonal Autoregressive Integrated Moving Average) model to forecast the demand.
It utilizes the pmdarima library to automatically select the optimal SARIMA parameters using the AIC (Akaike Information Criterion).
The SARIMA model is fitted to the historical demand data, and a forecast is generated for future periods.
The script plots the historical demand data, the fitted SARIMA model, and the forecasted values.

## Conclusion:
The script calculates the forecast accuracy metrics, such as MAPE (Mean Absolute Percentage Error), for the current forecast and the new forecast generated by the SARIMA model.
It provides a summary of the findings and key takeaways from the project, including the improvement in forecast accuracy achieved by using the SARIMA model.
It suggests exploring alternative methods, such as Facebook Prophet or XGBoost, to improve further the baseline established by SARIMA.
Overall, this code demonstrates a comprehensive approach to demand forecasting, encompassing data preprocessing, exploratory analysis, model building, and evaluation. It provides a framework for understanding and predicting future demand patterns based on historical data.

## Dependencies:
The dependencies for this project would typically include the following Python libraries:

- Pandas: Used for data manipulation and analysis
- Matplotlib: Used for data visualization and plotting
- Seaborn: Another data visualization library that provides a high-level interface for creating attractive and informative statistical graphics
- Statsmodels: A library that provides statistical models and functions for time series analysis, including the SARIMA model
- pmdarima: A library that provides an interface to the auto_arima function for automatically selecting optimal SARIMA parameters
- NumPy: A fundamental package for scientific computing with Python, used for numerical operations
- datetime: A module that supplies classes for working with dates and times
- sklearn.metrics: Part of the scikit-learn library, used for calculating forecast accuracy metrics like MAPE

