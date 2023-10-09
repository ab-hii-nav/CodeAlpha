import pandas as pd #To create pandas data frames(which are presented in stuctured data)
import numpy as np #numpy is used for numerical array selection
import matplotlib.pyplot as plt #To visualise data in graphical form
import seaborn as sns # To visualise data in graphical form

from sklearn.model_selection import train_test_split #to split dataset in to training and testing
from sklearn.linear_model import LogisticRegression #model to be applied
from sklearn.metrics import accuracy_score #performance evaluation matrices

# Load historical stock price data (replace 'your_data.csv' with your dataset)
data = pd.read_csv('/content/Google_test_data.csv')
print(data)


# Select the 'Date' and 'Close' columns
data = data[['Date', 'Close']]
# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])
# Set 'Date' as the index
data.set_index('Date', inplace=True)
# Define a feature (e.g., number of days in the future for prediction)
forecast_days = 30
# Create a new column 'Prediction' shifted 'forecast_days' days into the future
data['Prediction'] = data['Close'].shift(-forecast_days)
# Drop rows with missing data (NaN) in the 'Prediction' column
data.dropna(inplace=True)

# Create X (features) and y (target)
X = np.array(data.drop(['Prediction'], axis=1))
y = np.array(data['Prediction'])
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X.shape, X_train.shape, X_test.shape)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plot the actual vs. predicted prices for the test set
plt.figure(figsize=(12, 6))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(data.index[-len(y_test):], predictions, label='Predicted Prices')
plt.legend()
plt.show()
