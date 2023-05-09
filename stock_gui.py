import tkinter as tk
from tkinter import ttk
import sys
import os
import numpy as np
import yfinance as yf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import stock_lib


class ConsoleOutput:
    def __init__(self, master):
        self.textbox = tk.Text(master, state='disabled')
        self.textbox.pack(fill='both', expand=True)

        sys.stdout = self

    def write(self, text):
        self.textbox.configure(state='normal')
        self.textbox.insert('end', text)
        self.textbox.configure(state='disabled')


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


def prepare_data2(data, window_size, offset_value):
    x1, x2, x3 = ([] for r in range(3))
    y = []
    for i in range(offset_value + window_size * 3, len(data) - 1):
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


# Splitting the data for LSTM model
def split_data(x_input, y_input, split_ratio):
    (x_train, x_test, y_train, y_test) = train_test_split(x_input, y_input, test_size=split_ratio, shuffle=False)
    return x_train, x_test, y_train, y_test


def list_industry_update(*args):
    combo_stock_industry.set('')
    list_industry = nasdaq.loc[(nasdaq['Sector'] == var_stock_sector.get())]
    list_industry = list_industry['Industry'].unique()
    combo_stock_industry['values'] = list_industry.tolist()


def fill_stock_name(event):
    item = tree_stock.item(tree_stock.focus())
    selected = item['values']
    input_box_stock_name.delete(0, 'end')
    input_box_stock_name.insert(0, selected[0])


def search():
    stock_sector = var_stock_sector.get()
    stock_industry = var_stock_industry.get()
    stock = nasdaq.loc[(nasdaq['Sector'] == stock_sector) & (nasdaq['Industry'] == stock_industry)]
    stock = stock[['Symbol', 'Name', 'Country', 'IPO Year']]
    tree_stock.delete(*tree_stock.get_children())
    for index, row in stock.iterrows():
        tree_stock.insert('', 'end', values=list(row))


def submit():
    # Load the input from gui
    stock_name = input_box_stock_name.get()
    delta_time = int(input_box_delta_time.get())
    flag_train_data = var_train_data.get()
    window_size = int(input_box_window_size.get())
    train_epochs = int(input_box_train_epochs.get())
    flag_lstm_algo = int(var_lstm_algo.get())

    # Load the stock market data from Yahoo Finance
    start_date = datetime.now() - timedelta(days=delta_time)
    end_date = datetime.now()
    raw_data = yf.download(stock_name, start=start_date, end=end_date)

    # Prepare technical indicators for LSTM model
    df = pd.DataFrame(raw_data[['Close', 'Volume']])
    df['EMA10'] = ta.ema(df['Close'], length=10)
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['RSI14'] = ta.rsi(df['Close'], length=10)
    # df.index = df.index.astype(str).str.split('T').str[0]
    # df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    # df.index = df.index.strftime('%d-%m-%Y')
    # df_merged = pd.merge(df, df1, how='outer', left_index=True, right_index=True)
    # df = df_merged.loc[df.index]
    df1, earning_next, earning_delta = stock_lib.get_earning_data(stock_name)

    if flag_train_data == 'Close':
        features = ['Close']
    elif flag_train_data == 'Close, Volume':
        features = ['Close', 'Volume']
    elif flag_train_data == 'Close, Volume, RSI14':
        features = ['Close', 'Volume', 'RSI14']
    elif flag_train_data == 'Close, Volume, EMA10, RSI14':
        features = ['Close', 'Volume', 'EMA10', 'RSI14']
    elif flag_train_data == 'Close, Volume, EMA10, SMA20, RSI14':
        features = ['Close', 'Volume', 'EMA10', 'SMA20', 'RSI14']

    # Prepare data for LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    if isinstance(flag_train_data, int):
        data = scaler.fit_transform(df)
    else:
        data = scaler.fit_transform(df[features])
    offset_value = 20  # to remove the NaN value of SMA20

    if flag_lstm_algo == 0:
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
        model.fit(x_train, y_train, epochs=train_epochs, batch_size=32, validation_data=(x_test, y_test),
                  verbose=2)

        # Prediction for past days
        future_days_p = window_size * 2
        last_window_p = np.array(data[-window_size * 2:-window_size])
        predicted_p = []
        for i in range(future_days_p):
            next_day = np.array(
                model.predict(last_window_p.reshape(1, last_window_p.shape[0], last_window_p.shape[1])))
            predicted_p.append(next_day)
            last_window_p = np.concatenate((last_window_p[-window_size + 1:, :], next_day), axis=0)

        # Inverse transform the predicted values to their original scale
        predicted_p = np.array(predicted_p)
        predicted_p = predicted_p.reshape(-1, data.shape[1])
        predicted_p = scaler.inverse_transform(predicted_p)

        # Prediction for future days
        future_days_f = window_size
        last_window_f = np.array(data[-window_size:])
        predicted_f = []
        for i in range(future_days_f):
            next_day = np.array(
                model.predict(last_window_f.reshape(1, last_window_f.shape[0], last_window_f.shape[1])))
            predicted_f.append(next_day)
            last_window_f = np.concatenate((last_window_f[-window_size + 1:, :], next_day), axis=0)

        # Inverse transform the predicted values to their original scale
        predicted_f = np.array(predicted_f)
        predicted_f = predicted_f.reshape(-1, data.shape[1])
        predicted_f = scaler.inverse_transform(predicted_f)

        # Print the predicted values
        print(f'Past predicted closing: {predicted_p[:, 0]}')
        print(f'Future predicted closing: {predicted_f[:, 0]}')
        fig, ax = plt.subplots()
        ax.plot(np.arange(-window_size + 1, window_size + 1), predicted_p[:, 0], color='red',
                label='Predicted Past Closing')
        ax.plot(np.arange(1, window_size + 1), predicted_f[:, 0], color='orange', label='Predicted Future Closing')
        ax.plot(np.arange(-future_days_p + 1, 1), raw_data[['Close']][-future_days_p:], color='blue',
                label='Actual Closing')
        ax.plot(np.arange(-future_days_p + 1, 1), df['EMA10'][-future_days_p:], color='green', label='EMA10')
        ax.plot(np.arange(-future_days_p + 1, 1), df['SMA20'][-future_days_p:], color='cyan', label='SMA20')
        if earning_delta <= window_size:
            plt.axvline(x=earning_delta, color='red')
        ax2 = ax.twinx()
        ax2.set_ylim([0, 100])
        ax2.set_zorder(-100)
        ax.set_facecolor('none')
        ax2.scatter(np.arange(-future_days_p + 1, 1), df[['RSI14']][-future_days_p:], color='magenta', marker='x',
                    s=10, label='RSI14')
        plt.title(f"Predicted Market Price for {stock_name} in past {window_size} days")
        ax.legend()
        ax2.legend()
        plt.grid()

    elif flag_lstm_algo == 1:
        # Define function to create merged-LSTM-DNN model
        x1, x2, x3, y1 = prepare_data2(data, window_size, offset_value)
        (x1_train, x1_test, y1_train, y1_test) = split_data(x1, y1, 0.2)
        (x2_train, x2_test, y1_train, y1_test) = split_data(x2, y1, 0.2)
        (x3_train, x3_test, y1_train, y1_test) = split_data(x3, y1, 0.2)

        input1 = Input(shape=(x1_train.shape[1], x1_train.shape[2]))
        input2 = Input(shape=(x2_train.shape[1], x2_train.shape[2]))
        input3 = Input(shape=(x3_train.shape[1], x3_train.shape[2]))

        lstm11 = LSTM(64, return_sequences=True)(input1)
        lstm12 = LSTM(64, return_sequences=True)(input2)
        lstm13 = LSTM(64, return_sequences=True)(input3)

        lstm21 = LSTM(64, return_sequences=False)(lstm11)
        lstm22 = LSTM(64, return_sequences=False)(lstm12)
        lstm23 = LSTM(64, return_sequences=False)(lstm13)

        combined_lstm = concatenate([lstm21, lstm22, lstm23])
        d1 = Dense(75, activation='relu')(combined_lstm)
        d1 = Dropout(0.2)(d1)
        d2 = Dense(25, activation='relu')(d1)
        d2 = Dropout(0.2)(d2)
        output = Dense(data.shape[1])(d2)

        model = keras.Model(inputs=[input1, input2, input3], outputs=output)
        model.summary()
        model.compile(loss='mse', optimizer='adam')
        model.fit([x1_train, x2_train, x3_train], y1_train, epochs=train_epochs, batch_size=32,
                  validation_data=([x1_test, x2_test, x3_test], y1_test), verbose=2)

        # Prediction for past days
        future_days_p = window_size * 2
        last_window_p1 = np.array(data[-window_size * 2:-window_size])
        last_window_p2 = np.array(data[-window_size * 3:-window_size])
        last_window_p3 = np.array(data[-window_size * 4:-window_size])
        predicted_p = []
        for i in range(future_days_p):
            next_day = np.array(model.predict(
                [last_window_p1.reshape(1, last_window_p1.shape[0], last_window_p1.shape[1]),
                 last_window_p2.reshape(1, last_window_p2.shape[0], last_window_p2.shape[1]),
                 last_window_p3.reshape(1, last_window_p3.shape[0], last_window_p3.shape[1])]))
            predicted_p.append(next_day)
            last_window_p1 = np.concatenate((last_window_p1[-window_size + 1:, :], next_day), axis=0)
            last_window_p2 = np.concatenate((last_window_p2[-window_size * 2 + 1:, :], next_day), axis=0)
            last_window_p3 = np.concatenate((last_window_p3[-window_size * 3 + 1:, :], next_day), axis=0)

        # Inverse transform the predicted values to their original scale
        predicted_p = np.array(predicted_p)
        predicted_p = predicted_p.reshape(-1, data.shape[1])
        predicted_p = scaler.inverse_transform(predicted_p)

        # Prediction for future days
        future_days_f = window_size
        last_window_f1 = np.array(data[-window_size:])
        last_window_f2 = np.array(data[-window_size * 2:])
        last_window_f3 = np.array(data[-window_size * 3:])
        predicted_f = []
        for i in range(future_days_f):
            next_day = np.array(model.predict(
                [last_window_f1.reshape(1, last_window_f1.shape[0], last_window_f1.shape[1]),
                 last_window_f2.reshape(1, last_window_f2.shape[0], last_window_f2.shape[1]),
                 last_window_f3.reshape(1, last_window_f3.shape[0], last_window_f3.shape[1])]))
            predicted_f.append(next_day)
            last_window_f1 = np.concatenate((last_window_f1[-window_size + 1:, :], next_day), axis=0)
            last_window_f2 = np.concatenate((last_window_f2[-window_size * 2 + 1:, :], next_day), axis=0)
            last_window_f3 = np.concatenate((last_window_f3[-window_size * 3 + 1:, :], next_day), axis=0)

        # Inverse transform the predicted values to their original scale
        predicted_f = np.array(predicted_f)
        predicted_f = predicted_f.reshape(-1, data.shape[1])
        predicted_f = scaler.inverse_transform(predicted_f)

        # Print the predicted values
        print(f'Past predicted closing: {predicted_p[:, 0]}')
        print(f'Future predicted closing: {predicted_f[:, 0]}')
        fig, ax = plt.subplots()
        ax.plot(np.arange(-window_size + 1, window_size + 1), predicted_p[:, 0], color='red',
                label='Predicted Past Closing')
        ax.plot(np.arange(1, window_size + 1), predicted_f[:, 0], color='orange', label='Predicted Future Closing')
        ax.plot(np.arange(-future_days_p + 1, 1), raw_data[['Close']][-future_days_p:], color='blue',
                label='Actual Closing')
        ax.plot(np.arange(-future_days_p + 1, 1), df['EMA10'][-future_days_p:], color='green', label='EMA10')
        ax.plot(np.arange(-future_days_p + 1, 1), df['SMA20'][-future_days_p:], color='cyan', label='SMA20')
        if earning_delta <= window_size:
            plt.axvline(x=earning_delta, color='red')
        ax2 = ax.twinx()
        ax2.set_ylim([0, 100])
        ax2.set_zorder(-100)
        ax.set_facecolor('none')
        ax2.scatter(np.arange(-future_days_p + 1, 1), df[['RSI14']][-future_days_p:], color='magenta', marker='x',
                    s=10, label='RSI14')
        plt.title(f"Predicted Market Price for {stock_name} in past {window_size} days")
        ax.legend()
        ax2.legend()
        plt.grid()

    filename = f"{end_date.year}.{end_date.month}.{end_date.day}_{stock_name}_{flag_lstm_algo}.png"
    folder = f"/PyCharm/2023.04_StockPredict_LSTM/prt_scr/{stock_name}"
    if not os.path.exists(folder):
        # If it doesn't exist, create the directory
        os.makedirs(folder)
    plt.savefig(folder + "/" + filename)
    plt.show()


# Create GUI window
gui = tk.Tk()
gui.title('LSTM Stock Prediction')
gui.geometry("600x560")

# Create stock list from csv
nasdaq = pd.read_csv('nasdaq_screener_1680942591745.csv')
list_sector = nasdaq['Sector'].unique()
list_sector = list_sector.tolist()
list_industry = []

# Create label for the input box
stock_sector_label = tk.Label(gui, text="Stock Sector").place(x=20, y=20)
stock_industry_label = tk.Label(gui, text="Stock Industry").place(x=20, y=50)
stock_name_label = tk.Label(gui, text="Stock Name").place(x=20, y=80)
delta_time_label = tk.Label(gui, text="Data Period").place(x=20, y=110)
train_data_label = tk.Label(gui, text="Training Data").place(x=20, y=140)
window_size_label = tk.Label(gui, text="Window Size").place(x=20, y=170)
train_epochs_label = tk.Label(gui, text="Train Epochs").place(x=20, y=200)

# Create combo box for search func
var_stock_sector = tk.StringVar()
combo_stock_sector = ttk.Combobox(gui, values=list_sector, textvariable=var_stock_sector, width=57)
combo_stock_sector['state'] = 'readonly'
combo_stock_sector.pack
combo_stock_sector.place(x=110, y=20)

var_stock_sector.trace('w', list_industry_update)

var_stock_industry = tk.StringVar()
combo_stock_industry = ttk.Combobox(gui, values=list_industry, textvariable=var_stock_industry, width=57)
combo_stock_industry['state'] = 'readonly'
combo_stock_industry.pack
combo_stock_industry.place(x=110, y=50)

# Create input box for train func
input_box_stock_name = tk.Entry(gui, width=60)
input_box_stock_name.place(x=110, y=80)

input_box_delta_time = tk.Entry(gui, width=60)
input_box_delta_time.insert(tk.END, '1095')
input_box_delta_time.place(x=110, y=110)

input_box_window_size = tk.Entry(gui, width=60)
input_box_window_size.insert(tk.END, '100')
input_box_window_size.place(x=110, y=170)

input_box_train_epochs = tk.Entry(gui, width=60)
input_box_train_epochs.insert(tk.END, '1000')
input_box_train_epochs.place(x=110, y=200)

# Create checkbox
var_lstm_algo = tk.IntVar()
# Button_lstm_algo = tk.Checkbutton(gui, text="CAT.LSTM",
#                                   variable=var_lstm_algo,
#                                   onvalue=1,
#                                   offvalue=0,
#                                   height=2,
#                                   width=10,
#                                   anchor="w")
# Button_lstm_algo.place(x=500, y=530)
#
# Button_past_future_b = tk.Checkbutton(gui, text="Last 30 days",
#                                       variable=Checkbutton_past_future_b,
#                                       onvalue=1,
#                                       offvalue=0,
#                                       height=2,
#                                       width=10,
#                                       anchor="w")
# Button_past_future_b.place(x=320, y=50)

# Create a drop-down manual
list_train_data = ['Close',
                   'Close, Volume',
                   'Close, Volume, RSI14',
                   'Close, Volume, EMA10, RSI14',
                   'Close, Volume, EMA10, SMA20, RSI14']
var_train_data = tk.StringVar()
combo_train_data = ttk.Combobox(gui, textvariable=var_train_data, width=57)
combo_train_data['values'] = list_train_data
combo_train_data['state'] = 'readonly'
combo_train_data.current(4)
combo_train_data.pack
combo_train_data.place(x=110, y=140)

# Create radiobutton
var_lstm_algo = tk.IntVar()
RB_past_future_a = tk.Radiobutton(gui, text="DNN.LSTM", variable=var_lstm_algo, value=0)
RB_past_future_a.pack(anchor="w")
RB_past_future_a.place(x=500, y=20)
RB_past_future_b = tk.Radiobutton(gui, text="CAT.LSTM", variable=var_lstm_algo, value=1)
RB_past_future_b.pack(anchor="w")
RB_past_future_b.place(x=500, y=50)
RB_past_future_c = tk.Radiobutton(gui, text="Batch", variable=var_lstm_algo, value=2)
RB_past_future_c.pack(anchor="w")
RB_past_future_c.place(x=500, y=80)

# Create treeview widget
column = ('Symbol', 'Name', 'Country', 'IPO Year')
tree_stock = ttk.Treeview(gui, columns=column, show='headings')
for i in column:
    tree_stock.heading(i, text=i)
tree_stock.bind('<Double-1>', fill_stock_name)
tree_stock.pack
tree_stock.place(x=20, y=230, width=560, height=300)

# Create button
search_button = tk.Button(gui, text="Search", command=search, height=2, width=10).place(x=500, y=120)
submit_button = tk.Button(gui, text="Submit", command=submit, height=2, width=10).place(x=500, y=170)

# Run the GUI
gui.mainloop()
