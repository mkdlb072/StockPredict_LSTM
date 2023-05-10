import tensorflow as tf
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pytz
import os
from datetime import datetime
import pandas_ta as ta
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate, Add
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Load the stock market data from Yahoo Finance
def get_stock_data(stock_symbol, start_date, end_date):
    hist = yf.download(stock_symbol, start=start_date, end=end_date)
    hist = hist.drop('Adj Close', axis=1)
    hist.insert(0, 'Prev Close', hist['Close'].shift(1))
    # Calculate the technical indicators
    hist['EMA10'] = ta.ema(hist['Close'], length=10)
    hist['SMA20'] = ta.sma(hist['Close'], length=20)
    hist['RSI14'] = ta.rsi(hist['Close'], length=14)
    a = ta.adx(hist['High'], hist['Low'], hist['Close'], length=14)
    hist = hist.join(a)
    b = ta.aroon(hist['High'], hist['Low'], length=14)
    hist = hist.join(b['AROONOSC_14'])
    # Format the time in index
    hist.index = hist.index.astype(str).str.split('T').str[0]
    hist.index = pd.to_datetime(hist.index, format='%Y-%m-%d')
    hist.index = hist.index.strftime('%d-%m-%Y')
    hist = hist.iloc[(hist.isna().sum()).max():]
    return hist


def get_earning_data(stock_name):
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
    df = pd.DataFrame(
        {'Earnings_date': earnings_date, 'EPS_estimate': earnings_EPS_estimate,
         'EPS_reported': earnings_EPS_reported})
    df.set_index('Earnings_date', inplace=True)
    df.index = df.index.astype(str).str.split(',').str[:2].str.join(',')
    df.index = pd.to_datetime(df.index, format='%b %d, %Y')
    df.index = df.index.strftime('%d-%m-%Y')
    df = df[~df.index.duplicated(keep='first')]

    # Find earliest future date
    today = datetime.today()
    future_earning = df.index[df['EPS_reported'] == '-'].tolist()
    future_earning = [datetime.strptime(date_str, '%d-%m-%Y') for date_str in future_earning]
    future_dates = [date for date in future_earning if date >= today]
    if len(future_dates) != 0:
        earning_next = min(future_dates).strftime('%d-%m-%Y')
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        us_eastern = pytz.timezone('US/Eastern')
        today = datetime.now(tz=us_eastern)
        future_business_days = pd.date_range(start=today, periods=365, freq=us_bd)
        future_business_days = [d.strftime('%d-%m-%Y') for d in future_business_days]
        earning_delta = future_business_days.index(earning_next) + 1
    else:
        earning_next = '-'
        earning_delta = 365

    return df, earning_next, earning_delta


def prepare_data(data, window_size):
    x = []
    y = []
    for i in range(len(data) - window_size - 1):
        window = data[i:(i + window_size), :]
        x.append(window)
        y.append(data[i + window_size, :])
    x = np.array(x)
    y = np.array(y)
    return x, y


def prepare_data_3d(data, window_size):
    x1, x2, x3 = ([] for r in range(3))
    y = []
    for i in range(window_size * 3, len(data) - 1):
        w1 = data[i - window_size:i, :]
        x1.append(w1)
        w2 = data[i - window_size * 2:i, :]
        x2.append(w2)
        w3 = data[i - window_size * 3:i, :]
        x3.append(w3)
        y.append(data[i, :])
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    y = np.array(y)
    return x1, x2, x3, y


def train_test_split_3d(x1, x2, x3, y):
    (x1_train, x1_test, y1_train, y1_test) = train_test_split(x1, y, test_size=0.2, random_state=69)
    (x2_train, x2_test, y1_train, y1_test) = train_test_split(x2, y, test_size=0.2, random_state=69)
    (x3_train, x3_test, y1_train, y1_test) = train_test_split(x3, y, test_size=0.2, random_state=69)
    return x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test


def reform_transform(x, scaler):
    num_instances, num_time_steps, num_features = x.shape
    x = np.reshape(x, newshape=(-1, num_features))
    x = scaler.transform(x)
    x = np.reshape(x, newshape=(num_instances, num_time_steps, num_features))
    return x


def minmaxscale_3d(x1, x2, y1, y2):
    scaler = MinMaxScaler()
    # Transform x1
    num_instances, num_time_steps, num_features = x1.shape
    x1 = np.reshape(x1, newshape=(-1, num_features))
    x1 = scaler.fit_transform(x1)
    x1 = np.reshape(x1, newshape=(num_instances, num_time_steps, num_features))
    # Transform x2
    x2 = reform_transform(x2, scaler)
    # Transform y1, y2
    y1 = scaler.transform(y1)
    y2 = scaler.transform(y2)
    return x1, x2, y1, y2, scaler


def minmaxscale_4d(x11, x12, x21, x22, x31, x32, y1, y2):
    scaler = MinMaxScaler()
    # Transform merged inputs x_train
    num_instances, num_time_steps, num_features = x31.shape
    x31 = np.reshape(x31, newshape=(-1, num_features))
    x31 = scaler.fit_transform(x31)
    x31 = np.reshape(x31, newshape=(num_instances, num_time_steps, num_features))
    x32 = reform_transform(x32, scaler)
    x21 = reform_transform(x21, scaler)
    x22 = reform_transform(x22, scaler)
    x11 = reform_transform(x11, scaler)
    x12 = reform_transform(x12, scaler)
    # Transform y1, y2
    y1 = scaler.transform(y1)
    y2 = scaler.transform(y2)
    return x11, x12, x21, x22, x31, x32, y1, y2, scaler


def define_lstm_cnn_model(x_train):
    # Define the CNN model
    cnn_model = tf.keras.Sequential([
        layers.Conv1D(32, kernel_size=1, activation='sigmoid', padding="same",
                      input_shape=(x_train.shape[1], x_train.shape[2])),
        layers.MaxPooling1D(pool_size=1, padding="same"),
    ])
    # Define the bi-lstm model
    lstm_model = tf.keras.Sequential([
        layers.Bidirectional(layers.LSTM(64, activation='tanh')),
        layers.Dense(x_train.shape[2]),
    ])
    # Define the combined model
    inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = cnn_model(inputs)
    outputs = lstm_model(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def define_lstm_dnn_model(x_train):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dropout(0.2))
    model.add(Dense(x_train.shape[2]))
    return model


def define_lstm_cat_model(x1_train, x2_train, x3_train):
    input1 = Input(shape=(x1_train.shape[1], x1_train.shape[2]))
    input2 = Input(shape=(x2_train.shape[1], x2_train.shape[2]))
    input3 = Input(shape=(x3_train.shape[1], x3_train.shape[2]))
    # Define first lstm layer
    lstm11 = LSTM(64, return_sequences=True)(input1)
    lstm12 = LSTM(64, return_sequences=True)(input2)
    lstm13 = LSTM(64, return_sequences=True)(input3)
    # Define second lstm layer
    lstm21 = LSTM(64, return_sequences=False)(lstm11)
    lstm22 = LSTM(64, return_sequences=False)(lstm12)
    lstm23 = LSTM(64, return_sequences=False)(lstm13)
    # Concatenate all lstm layer
    combined_lstm = concatenate([lstm21, lstm22, lstm23])
    d1 = Dense(75, activation='relu')(combined_lstm)
    d1 = Dropout(0.2)(d1)
    d2 = Dense(25, activation='relu')(d1)
    d2 = Dropout(0.2)(d2)
    output = Dense(x1_train.shape[2])(d2)
    model = keras.Model(inputs=[input1, input2, input3], outputs=output)
    return model


def train_save_model(model, x_train, y_train, x_test, y_test, epoch, flg_new_model, stock_symbol, dev_env):
    if dev_env == "PC":
        save_path = f'/PyCharm/2023.04_StockPredict_LSTM/model/{stock_symbol}'
    else:
        save_path = f'/Git/StockPredict_LSTM/model/{stock_symbol}'
    if flg_new_model == 1:
        # Train the model
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=epoch, batch_size=32, validation_data=(x_test, y_test), verbose=2)
        # Save the model
        today = datetime.now()
        # Define path to save model
        if not os.path.exists(save_path):
            # If it doesn't exist, create the directory
            os.makedirs(save_path)
        model.save(os.path.join(save_path, f'{today.year}.{today.month}.{today.day}_{stock_symbol}.h5'))
    else:
        model_path = max(os.listdir(save_path), key=os.path.getctime)
        model = tf.keras.models.load_model(os.path.join(save_path, model_path))
    return model


def get_past_prediction(model, data, pred_window, dt_window, scaler):
    last_window = np.array(data[-pred_window - dt_window:-pred_window])
    last_window = scaler.transform(last_window)
    prediction = []
    for i in range(pred_window * 2):
        next_day = np.array(model.predict(last_window.reshape(1, last_window.shape[0], last_window.shape[1])))
        prediction.append(next_day)
        last_window = np.concatenate((last_window[-dt_window + 1:, :], next_day), axis=0)

    # Inverse transform the predicted values to their original scale
    prediction = np.array(prediction)
    prediction = prediction.reshape(-1, data.shape[1])
    prediction = scaler.inverse_transform(prediction)
    return prediction


def get_past_prediction_3d(model, data, pred_window, scaler):
    last_window_p1 = np.array(data[-pred_window * 2:-pred_window])
    last_window_p2 = np.array(data[-pred_window * 3:-pred_window])
    last_window_p3 = np.array(data[-pred_window * 4:-pred_window])
    prediction = []
    for i in range(pred_window * 2):
        next_day = np.array(model.predict(
            [last_window_p1.reshape(1, last_window_p1.shape[0], last_window_p1.shape[1]),
             last_window_p2.reshape(1, last_window_p2.shape[0], last_window_p2.shape[1]),
             last_window_p3.reshape(1, last_window_p3.shape[0], last_window_p3.shape[1])]))
        prediction.append(next_day)
        last_window_p1 = np.concatenate((last_window_p1[-pred_window + 1:, :], next_day), axis=0)
        last_window_p2 = np.concatenate((last_window_p2[-pred_window * 2 + 1:, :], next_day), axis=0)
        last_window_p3 = np.concatenate((last_window_p3[-pred_window * 3 + 1:, :], next_day), axis=0)

    # Inverse transform the predicted values to their original scale
    prediction = np.array(prediction)
    prediction = prediction.reshape(-1, data.shape[1])
    prediction = scaler.inverse_transform(prediction)
    return prediction


def get_future_prediction(model, data, pred_window, dt_window, scaler):
    last_window = np.array(data[-dt_window:])
    last_window = scaler.transform(last_window)
    prediction = []
    for i in range(pred_window):
        next_day = np.array(model.predict(last_window.reshape(1, last_window.shape[0], last_window.shape[1])))
        prediction.append(next_day)
        last_window = np.concatenate((last_window[-dt_window + 1:, :], next_day), axis=0)

        # Inverse transform the predicted values to their original scale
    prediction = np.array(prediction)
    prediction = prediction.reshape(-1, data.shape[1])
    prediction = scaler.inverse_transform(prediction)
    return prediction


def get_future_prediction_3d(model, data, pred_window, scaler):
    last_window_f1 = np.array(data[-pred_window:])
    last_window_f2 = np.array(data[-pred_window * 2:])
    last_window_f3 = np.array(data[-pred_window * 3:])
    prediction = []
    for i in range(pred_window):
        next_day = np.array(model.predict(
            [last_window_f1.reshape(1, last_window_f1.shape[0], last_window_f1.shape[1]),
             last_window_f2.reshape(1, last_window_f2.shape[0], last_window_f2.shape[1]),
             last_window_f3.reshape(1, last_window_f3.shape[0], last_window_f3.shape[1])]))
        prediction.append(next_day)
        last_window_f1 = np.concatenate((last_window_f1[-pred_window + 1:, :], next_day), axis=0)
        last_window_f2 = np.concatenate((last_window_f2[-pred_window * 2 + 1:, :], next_day), axis=0)
        last_window_f3 = np.concatenate((last_window_f3[-pred_window * 3 + 1:, :], next_day), axis=0)

    # Inverse transform the predicted values to their original scale
    prediction = np.array(prediction)
    prediction = prediction.reshape(-1, data.shape[1])
    prediction = scaler.inverse_transform(prediction)
    return prediction


def plot_prediction(stock_symbol, raw_data, predict_window, predicted_p, predicted_f, earning_delta):
    fig, ax = plt.subplots()
    ax.plot(np.arange(-predict_window + 1, predict_window + 1), predicted_p[:, 0], color='red',
            label='Predicted Past Closing')
    ax.plot(np.arange(1, predict_window + 1), predicted_f[:, 0], color='orange', label='Predicted Future Closing')
    ax.plot(np.arange(-predict_window * 2 + 1, 1), raw_data[['Close']][-predict_window * 2:], color='blue',
            label='Actual Closing')
    ax.plot(np.arange(-predict_window * 2 + 1, 1), raw_data['EMA10'][-predict_window * 2:], color='green',
            label='EMA10')
    ax.plot(np.arange(-predict_window * 2 + 1, 1), raw_data['SMA20'][-predict_window * 2:], color='cyan', label='SMA20')
    if earning_delta <= predict_window:
        plt.axvline(x=earning_delta, color='red')
    ax2 = ax.twinx()
    ax2.set_ylim([0, 100])
    ax2.set_zorder(-100)
    ax.set_facecolor('none')
    ax2.scatter(np.arange(-predict_window * 2 + 1, 1), raw_data[['RSI14']][-predict_window * 2:], color='magenta',
                marker='x',
                s=10, label='RSI14')
    plt.title(f"Predicted Market Price for {stock_symbol} in {predict_window} days")
    ax.legend()
    ax2.legend()
    plt.grid()


def save_graph(today, stock_symbol, flg_lstm_algo, dev_env):
    filename = f'{today.year}.{today.month}.{today.day}_{stock_symbol}_{flg_lstm_algo}.png'
    if dev_env == "PC":
        folder = f'/PyCharm/2023.04_StockPredict_LSTM/prt_scr/{stock_symbol}'
    else:
        folder = f'/Git/StockPredict_LSTM/prt_scr/{stock_symbol}'
    if not os.path.exists(folder):
        # If it doesn't exist, create the directory
        os.makedirs(folder)
    plt.savefig(folder + "/" + filename)
    plt.show()
