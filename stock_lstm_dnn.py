# Import necessary libraries
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pytz

# Load the stock market data from Yahoo Finance
delta_time = 1095
start_date = datetime.now() - timedelta(days=delta_time)
end_date = datetime.now()
stock_name = 'ENPH'
raw_data = yf.download(stock_name, start=start_date, end=end_date)


# Define function to prepare data for LSTM model
def prepare_data(data, window_size, offset_value):
    x = []
    y = []
    for i in range(offset_value, len(data) - window_size - 1):
        window = data[i:(i + window_size), :]
        x.append(window)
        y.append(data[i + window_size, :])
    x = np.array(x)
    y = np.array(y)
    return x, y


# Splitting the data for LSTM model
def split_data(x_input, y_input, split_ratio):
    (x_train, x_test, y_train, y_test) = train_test_split(x_input, y_input, test_size=split_ratio, shuffle=False)
    return x_train, x_test, y_train, y_test

# Send a GET request to the page and get the HTML content
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
# Set the URL of the earnings page for the stock you're interested in
url = f'https://finance.yahoo.com/calendar/earnings/?symbol={stock_name}'
response = requests.get(url, headers=headers)
html_content = response.content

# Use BeautifulSoup to parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Extract the text values of the earnings date and result elements
earnings_date = []
earnings_EPS_estimate = []
earnings_EPS_reported = []
for tag in soup.find_all('td', {'aria-label': 'Earnings Date'}):
    earnings_date.append(tag.text)
for tag in soup.find_all('td', {'aria-label': 'EPS Estimate'}):
    earnings_EPS_estimate.append(tag.text)
for tag in soup.find_all('td', {'aria-label': 'Reported EPS'}):
    earnings_EPS_reported.append(tag.text)

# Prepare the dataframe for time alignment
df1 = pd.DataFrame(
    {'Earnings_date': earnings_date, 'EPS_estimate': earnings_EPS_estimate, 'EPS_reported': earnings_EPS_reported})
df1.set_index('Earnings_date', inplace=True)
df1.index = df1.index.astype(str).str.split(',').str[:2].str.join(',')
df1.index = pd.to_datetime(df1.index, format='%b %d, %Y')
df1.index = df1.index.strftime('%d-%m-%Y')
df1 = df1[~df1.index.duplicated(keep='first')]

# Find the day of future earnings and deltatime
future_earning = df1.index[df1['EPS_reported'] == '-'].tolist()
future_earning = [datetime.strptime(date_str, '%d-%m-%Y') for date_str in future_earning]
future_dates = [date for date in future_earning if date >= end_date]
if len(future_dates)!= 0:
    next_earning = min(future_dates).strftime('%d-%m-%Y')
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    us_eastern = pytz.timezone('US/Eastern')
    today = datetime.now(tz=us_eastern)
    future_business_days = pd.date_range(start=today, periods=365, freq=us_bd)
    future_business_days = [d.strftime('%d-%m-%Y') for d in future_business_days]
    earning_delta = future_business_days.index(next_earning) + 1
else:
    earning_delta = 365

# Prepare technical indicators for LSTM model
df = pd.DataFrame(raw_data[['Close', 'Volume']])
df['RSI10'] = ta.rsi(df['Close'], length=10)
df['EMA10'] = ta.ema(df['Close'], length=10)
df['SMA20'] = ta.sma(df['Close'], length=20)
df.index = df.index.astype(str).str.split('T').str[0]
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df.index = df.index.strftime('%d-%m-%Y')
df_merged = pd.merge(df, df1, how='outer', left_index=True, right_index=True)
df = df_merged.loc[df.index]

# Prepare data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df)
window_size = 30
offset_value = 20  # to remove the NaN value of SMA20
x, y = prepare_data(data, window_size, offset_value)
(x_train, x_test, y_train, y_test) = split_data(x, y, 0.3)


# Define function to create LSTM-DNN model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dropout(0.2))
model.add(Dense(data.shape[1]))
model.summary()


# Compile and fit model
model.compile(loss='mse', optimizer='adam')
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=2)


# Predict future stock values
future_days = 60
last_window = np.array(data[-window_size * 2:-window_size])
predicted = []
for i in range(future_days):
    next_day = np.array(model.predict(last_window.reshape(1, last_window.shape[0], last_window.shape[1])))
    predicted.append(next_day)
    last_window = np.concatenate((last_window[-window_size+1:, :], next_day), axis=0)

# Inverse transform the predicted values to their original scale
predicted = np.array(predicted)
predicted = predicted.reshape(-1, data.shape[1])
predicted = scaler.inverse_transform(predicted)

# Print the predicted values
print(predicted[:, 0])

# Plot the results
plt.plot(np.arange(-29, 31), predicted[:, 0], color='red', label='Predicted Price')
plt.plot(np.arange(-59, 1), raw_data[['Close']][-window_size*2:], color='blue', label='Actual Price')
plt.title(f"Predicted Market Price for {stock_name} in coming 30 days")
plt.legend()
plt.grid()
plt.show()
