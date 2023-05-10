import tkinter as tk
from tkinter import ttk
import sys
import pandas as pd
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
    stock_symbol = input_box_stock_name.get()
    delta_time = int(input_box_delta_time.get())
    flg_train_data = var_train_data.get()
    data_window = int(input_box_window_size.get())
    epoch = int(input_box_train_epochs.get())
    flg_lstm_algo = int(var_lstm_algo.get())
    predict_window = 100
    flg_train_model = 1
    dev_env = 'NB'

    # Get stock and earning data from yFinance
    start_date = datetime.now() - timedelta(days=delta_time)
    end_date = datetime.now()
    raw_data = stock_lib.get_stock_data(stock_symbol, start_date, end_date)
    np_data = raw_data.to_numpy()
    earning_table, earning_next, earning_delta = stock_lib.get_earning_data(stock_symbol)

    # if flg_train_data == 'Close':
    #     features = ['Close']
    # elif flg_train_data == 'Close, Volume':
    #     features = ['Close', 'Volume']
    # elif flg_train_data == 'Close, Volume, RSI14':
    #     features = ['Close', 'Volume', 'RSI14']
    # elif flg_train_data == 'Close, Volume, EMA10, RSI14':
    #     features = ['Close', 'Volume', 'EMA10', 'RSI14']
    # elif flg_train_data == 'Close, Volume, EMA10, SMA20, RSI14':
    #     features = ['Close', 'Volume', 'EMA10', 'SMA20', 'RSI14']

    if flg_lstm_algo == 0:
        # Split the data for training and testing
        x, y = stock_lib.prepare_data(np_data, data_window)
        (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
        # Scaling the training data
        (x_train, x_test, y_train, y_test, sc) = stock_lib.minmaxscale_3d(x_train, x_test, y_train, y_test)
        # Train the model
        model = stock_lib.define_lstm_dnn_model(x_train)
        model = stock_lib.train_save_model(model, x_train, y_train, x_test, y_test, epoch, flg_train_model,
                                           stock_symbol, dev_env)
        predicted_p = stock_lib.get_past_prediction(model, np_data, predict_window, data_window, sc)
        predicted_f = stock_lib.get_future_prediction(model, np_data, predict_window, data_window, sc)
        print(predicted_p[:, 0])

    elif flg_lstm_algo == 1:
        # Split the data for lstm_cat training and testing
        x1, x2, x3, y = stock_lib.prepare_data_3d(np_data, predict_window)
        (x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test) = stock_lib.train_test_split_3d(x1,
                                                                                                                   x2,
                                                                                                                   x3,
                                                                                                                   y)
        # Scaling the training data
        (x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test, sc) = stock_lib.minmaxscale_4d(
            x1_train,
            x1_test,
            x2_train,
            x2_test,
            x3_train,
            x3_test,
            y_train,
            y_test)
        # Train the model
        model = stock_lib.define_lstm_cat_model(x1_train, x2_train, x3_train)
        model.summary()
        model = stock_lib.train_save_model(model, [x1_train, x2_train, x3_train], y_train, [x1_test, x2_test, x3_test],
                                           y_test, epoch, flg_train_model, stock_symbol, dev_env)
        predicted_p = stock_lib.get_past_prediction_3d(model, np_data, predict_window, sc)
        predicted_f = stock_lib.get_future_prediction_3d(model, np_data, predict_window, sc)
        print(predicted_p[:, 0])

    elif flg_lstm_algo == 2:
        # Split the data for training and testing
        x, y = stock_lib.prepare_data(np_data, data_window)
        (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
        # Scaling the training data
        (x_train, x_test, y_train, y_test, sc) = stock_lib.minmaxscale_3d(x_train, x_test, y_train, y_test)
        # Train the model
        model = stock_lib.define_lstm_cnn_model(x_train)
        model.summary()
        model = stock_lib.train_save_model(model, x_train, y_train, x_test, y_test, epoch, flg_train_model,
                                           stock_symbol, dev_env)
        predicted_p = stock_lib.get_past_prediction(model, np_data, predict_window, data_window, sc)
        predicted_f = stock_lib.get_future_prediction(model, np_data, predict_window, data_window, sc)
        print(predicted_p[:, 0])

    # Plot the results
    stock_lib.plot_prediction(stock_symbol, raw_data, predict_window, predicted_p, predicted_f, earning_delta)
    stock_lib.save_graph(end_date, stock_symbol, flg_lstm_algo, dev_env)


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
RB_past_future_c = tk.Radiobutton(gui, text="CNN.LSTM", variable=var_lstm_algo, value=2)
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
