import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
import sqlite3
import datetime

# Load the Bitcoin price data (You can use your own dataset)
# Example: df = pd.read_csv('bitcoin_price_data.csv')
# Assuming your dataset has a 'Close' column for Bitcoin prices
#prices = df['Close'].values.reshape(-1, 1)
# btc = yf.Ticker('BTC-USD')
# data = btc.history(period='5y')
# data.drop(columns=['Open', 'High', 'Low', 'Dividends', 'Stock Splits'], axis = 1, inplace=True)
# data.head()
conn = sqlite3.connect('Database.db')
data = pd.read_sql('SELECT Date, Close, Volume FROM PRICE', conn)
prices = data['Close'].values.reshape(-1,1)
# Normalize the data
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Split the data into training and testing sets
train_size = int(0.8 * len(prices_scaled))
train_data = prices_scaled[:train_size]
test_data = prices_scaled[train_size:len(prices_scaled)-1]

# Create sequences for training
sequence_length = 10  # Number of previous time steps to use as input
X_train, y_train = [], []
for i in range(sequence_length, len(train_data)):
    X_train.append(train_data[i - sequence_length:i])
    y_train.append(train_data[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Define an initial learning rate
initial_learning_rate = 0.001

# Create a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,  # Initial learning rate
    decay_steps=1000,      # Number of steps to decay the learning rate
    decay_rate=0.9,         # Rate of decay
    staircase=True          # Optional: Use a staircase decay
)

# Build the RNN model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Create sequences for testing
X_test, y_test = [], []
for i in range(sequence_length, len(test_data)):
    X_test.append(test_data[i - sequence_length:i])
    y_test.append(test_data[i])
X_test, y_test = np.array(X_test), np.array(y_test)

predictions = model.predict(X_test)
predictions = predictions.reshape(-1)  # Reshape to (n_samples,)

# Inverse scaling
predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))

#Plot the results
plt.figure(figsize=(12, 6))
plt.plot(prices[train_size + sequence_length:], label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.show()

data = pd.read_sql('SELECT Date, Close, Volume FROM PRICE LIMIT 100', conn)

# Prepare new input data (contains the most recent historical data)
new_data = data['Close'].values.reshape(-1,1)  # Your new dataset with the most recent historical data
new_data_scaled = scaler.transform(new_data.reshape(-1, 1))

# Create a single sequence for prediction
X_pred = new_data_scaled[-sequence_length:].reshape(1, sequence_length, 1)

# Make predictions for the future
future_predictions = []

for _ in range(10):  # Predict the next 10 days
    prediction = model.predict(X_pred)
    future_predictions.append(prediction[0, 0])  # Extract the prediction value
    # Shift the input sequence one step to the right and append the new prediction
    X_pred = np.roll(X_pred, shift=-1)
    X_pred[0, -1, 0] = prediction[0, 0]

# Inverse scaling for future predictions
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot the historical data and future predictions
plt.figure(figsize=(12, 6))
plt.plot(new_data, label='Historical Data')
plt.plot(range(len(new_data), len(new_data) + len(future_prices)), future_prices, label='Future Predictions', linestyle='--')
plt.legend()
plt.show()

print(future_predictions)
print(future_prices)






